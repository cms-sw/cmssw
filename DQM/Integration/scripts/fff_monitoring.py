#!/usr/bin/env python

import os
import sys
import logging
import re
import datetime
import subprocess
import socket
import time
import select
import json
import datetime

log = logging.getLogger(__name__)

def prepare_imports():
    # minihack
    sys.path.append('/opt/hltd/python')
    sys.path.append('/opt/hltd/lib')

    global inotify, watcher, es_client

    import _inotify as inotify
    import watcher
    import pyelasticsearch.client as es_client


class DQMMonitor(object):
    def __init__(self, top_path, rescan_timeout=30):
        self.path = top_path
        self.rescan_timeout = rescan_timeout
        self.es = es_client.ElasticSearch("http://127.0.0.1:9200")
        self.index_name = "dqm_online_monitoring"

        try:
            os.makedirs(self.path)
        except OSError:
            pass

        self.mask = inotify.IN_CLOSE_WRITE | inotify.IN_MOVED_TO
        self.w = watcher.Watcher()
        self.w.add(self.path, self.mask)

    def recreate_index(self):
        self.delete_index()
        self.create_index()

    def delete_index(self):
        log.info("Deleting index: %s", self.index_name)
        self.es.delete_index(self.index_name)

    def create_index(self):
        log.info("Creating index: %s", self.index_name)

        self.settings = {
            "analysis": {
                "analyzer": {
                    "prefix-test-analyzer": {
                        "type": "custom",
                        "tokenizer": "prefix-test-tokenizer"
                    }
                },
                "tokenizer": {
                    "prefix-test-tokenizer": {
                        "type": "path_hierarchy",
                        "delimiter": "_"
                    }
                }
            },
            "index":{
                'number_of_shards' : 16,
                'number_of_replicas' : 1
            }
        }

        self.mappings = {
            'dqm-source-state' : {
                'properties' : {
                    'type' : {'type' : 'string' },
                    'pid' : { 'type' : 'integer' },
                    'hostname' : { 'type' : 'string' },
                    'sequence' : { 'type' : 'integer', "index" : "not_analyzed" },
                    'run' : { 'type' : 'integer' },
                    'lumi' : { 'type' : 'integer' },
                },
                '_timestamp' : { 'enabled' : True, 'store' : True, },
                '_ttl' : { 'enabled' : True, 'default' : '15d' }
            },
            'dqm-diskspace' : {
                'properties' : {
                    'type' : {'type' : 'string' },
                    'pid' : { 'type' : 'integer' },
                    'hostname' : { 'type' : 'string' },
                    'sequence' : { 'type' : 'integer', "index" : "not_analyzed" },
                },
                '_timestamp' : { 'enabled' : True, 'store' : True, },
                '_ttl' : { 'enabled' : True, 'default' : '15d' }
            },
        }

        try:
            self.es.create_index(self.index_name, settings={ 'settings': self.settings, 'mappings': self.mappings })
        except es_client.IndexAlreadyExistsError:
            logger.info("Index already exists.", exc_info=True)
            pass
        except:
            logger.warning("Cannot create index", exc_info=True)

        log.info("Created index: %s", self.index_name)

    def upload_file(self, fp, preprocess=None):
        log.info("Uploading: %s", fp)

        try:
            f = open(fp, "r")
            document = json.load(f)
            f.close()

            if preprocess:
                document = preprocess(document)

            ret = self.es.index(self.index_name, document["type"], document, id=document["_id"])
        except:
            log.warning("Failure to upload the document: %s", fp, exc_info=True)

    def process_file(self, fp):
        fname = os.path.basename(fp)

        if fname.startswith("."):
            return

        if not fname.endswith(".jsn"):
            return

        self.upload_file(fp)
        os.unlink(fp)

    def process_dir(self):
        for f in os.listdir(self.path):
            fp = os.path.join(self.path, f)
            self.process_file(fp)

    def run(self):
        poll = select.poll()
        poll.register(self.w, select.POLLIN)
        poll.poll(self.rescan_timeout*1000)

        # clear the events
        for event in self.w.read(bufsize=0):
            pass
            #print event

        self.process_dir()

    def run_daemon(self):
        self.process_dir()

        while True:
            service.run()

    def run_playback(self, directory, scale=2):
        files = os.listdir(directory)
        todo = []
        for f in files:
            spl = f.split("+")
            if len(spl) < 2:
                continue

            date, seq = spl[0].split(".")
            date, seq = datetime.datetime.fromtimestamp(long(date)), long(seq)

            todo.append({'date': date, 'seq': seq, 'f': os.path.join(directory, f)})

        def ts(td):
            # because total_seconds() is missing in 2.6
            return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6

        todo.sort(key=lambda x: (x["date"], x["seq"], ))
        m = todo[0]["date"]
        for f in todo:
            f["diff"] = ts(f["date"] - m) / scale

        def hotfix(doc):
            doc["tag"] = os.path.basename(doc["tag"])
            return doc

        start_time = datetime.datetime.now()
        while todo:
            elapsed = ts(datetime.datetime.now() - start_time)
            if todo[0]["diff"] <= elapsed:
                item = todo.pop(0)
                self.upload_file(item["f"], preprocess=hotfix)
            else:
                time.sleep(0.2)

# use a named socket check if we are running
# this is very clean and atomic and leave no files
# from: http://stackoverflow.com/a/7758075
def lock(pname):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.bind('\0' + pname)
        return sock
    except socket.error:
        return None

# same as in fff_deleter.py
def daemonize(logfile, pidfile):
    # do the double fork
    pid = os.fork()
    if pid != 0:
        sys.exit(0)

    os.setsid()
    sys.stdin.close()
    sys.stdout.close()
    sys.stderr.close()

    fl = open(logfile, "a")
    sys.stdout = fl
    sys.stderr = fl

    pid = os.fork()
    if pid != 0:
        sys.exit(0)

    if pidfile:
        f = open(pidfile, "w")
        f.write("%d\n" % os.getpid())
        f.close()

if __name__ == "__main__":
    do_mode = "daemon"

    do_foreground = False
    if len(sys.argv) > 1 and sys.argv[1] == "reindex":
        do_mode = "reindex"
        do_foreground = True

    if len(sys.argv) > 1 and sys.argv[1] == "playback":
        do_mode = "playback"
        do_foreground = True

    if not do_foreground:
        # try to take the lock or quit
        sock = lock("fff_dqmmon")
        if sock is None:
            sys.stderr.write("Already running, exitting.\n")
            sys.stderr.flush()
            sys.exit(1)

        daemonize("/var/log/fff_monitoring.log", "/var/run/fff_monitoring.pid")

    # log to stderr (it might be redirected)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    flog_ch = logging.StreamHandler()
    flog_ch.setLevel(logging.INFO)
    flog_ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(flog_ch)

    # write the pid file
    log.info("Pid is %d", os.getpid())

    prepare_imports()

    service = DQMMonitor(
        top_path = "/tmp/dqm_monitoring/",
    )

    if do_mode == "reindex":
        service.recreate_index()
    elif do_mode == "playback":
        #service.recreate_index()
        service.run_playback(sys.argv[2])
    else:
        service.run_daemon()
