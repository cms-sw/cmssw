import re
import time
import uproot
import struct
import socket
import sqlite3
import tempfile
import subprocess

DBNAME = '../data/directory.sqlite'

DBSCHEMA = """
BEGIN;
CREATE TABLE IF NOT EXISTS samples(dataset, run, lumi, filename, menamesid, meoffsetsid);
CREATE INDEX IF NOT EXISTS samplelookup ON samples(dataset, run, lumi);
CREATE UNIQUE INDEX IF NOT EXISTS uniquesamples on samples(dataset, run, lumi, filename);
CREATE TABLE IF NOT EXISTS menames(menamesid INTEGER PRIMARY KEY, menameblob);
CREATE UNIQUE INDEX IF NOT EXISTS menamesdedup ON menames(menameblob);
CREATE TABLE IF NOT EXISTS meoffsets(meoffsetsid INTEGER PRIMARY KEY, meoffsetsblob);
COMMIT;
"""

with sqlite3.connect(DBNAME) as db:
    db.executescript(DBSCHEMA)

class DQMDataStore:

    def __init__(self):
        self.db = sqlite3.connect(DBNAME)

    def __del__(self):
        self.db.close()

    def __execute(self, sql, args=None):
        c = self.db.cursor()
        if args:
            return  c.execute(sql, args)
        else:
            return c.execute(sql)

    def get_samples(self, run, dataset):
        if run:
            run = '%%%s%%' % run
        if dataset:
            dataset = '%%%s%%' % dataset

        results = []

        if run == dataset == None:
            return samples
        elif run != None and dataset != None:
            sql = 'SELECT run, dataset FROM samples WHERE dataset LIKE ? AND run LIKE ?'
            results = self.__execute(sql, (dataset, run))
        elif run != None:
            sql = 'SELECT run, dataset FROM samples WHERE run LIKE ?'
            results = self.__execute(sql, (run,))
        elif dataset != None:
            sql = 'SELECT run, dataset FROM samples WHERE dataset LIKE ?'
            results = self.__execute(sql, (dataset,))

        return results
    
    def get_me_list_blob(self, run, dataset):
        """
        # One me is represented as a multiple of 2 lines in a list
        # First line contains and ME path and secnod line contains secondary string:
        # a metadada of that ME, if it exists.
        # Possible cases of metadata: efficiency flag or QTest
        # If one ME has multiple secondary strings, its name in the list will appear multiple times,
        # so lines can always be parsed 2 at a time.
        # If secondary string doesn't exist, second line will be empty. That is 99% of all the cases.
        # Example:
        # me1
        # me1_qtest1
        # me1
        # me1_qtest2
        """
        # For now adding a LIMIT 1 because there might be multiple version of the same file.
        sql = 'SELECT menameblob FROM samples JOIN menames ON samples.menamesid = menames.menamesid WHERE run = ? AND dataset = ? LIMIT 1;'
        blob = self.__execute(sql, (int(run), dataset))
        blob = list(blob)[0][0]
        return blob

    # def get_me_offsets_blob(self, run, dataset):
    #     # For now adding a LIMIT 1 because there might be multiple version of the same file.
    #     sql = 'SELECT meoffsetsblob FROM samples JOIN meoffsets ON samples.meoffsetsid = meoffsets.meoffsetsid WHERE run = ? AND dataset = ? LIMIT 1;'
    #     blob = self.__execute(sql, (int(run), dataset))
    #     blob = list(blob)[0][0]
    #     return blob

    def get_blobs_and_filename(self, run, dataset):
        # For now adding a LIMIT 1 because there might be multiple version of the same file.
        sql = '''
        SELECT
            filename,
            menameblob,
            meoffsetsblob
        FROM
            samples
            JOIN menames ON samples.menamesid = menames.menamesid
            JOIN meoffsets ON samples.meoffsetsid = meoffsets.meoffsetsid
        WHERE
            run = ?
        AND 
            dataset = ?
        LIMIT 1;
        '''
        cur = self.__execute(sql, (int(run), dataset))
        cur = list(cur)

        filename = cur[0][0]
        list_blob = cur[0][1]
        offsets_blob = cur[0][2]

        return (filename, list_blob, offsets_blob)


    def get_rendered_image(self, run, dataset, path, width=200, height=200):
        print(run, dataset, path)

        width = int(width)
        height = int(height)

        filename, list_blob, offsets_blob = self.get_blobs_and_filename(run, dataset)
        list_buf = zlib.decompress(list_blob)
        list_lines = list_buf.split(b'\n')
        offsets_buf = zlib.decompress(offsets_blob)
        # offsets_lines = offsets_buf.split(b'\n')

        # Find the index of run/dataset/me in me list blob. 
        # The index in me list will correspond to the index in offsets list.
        # TODO: use binary search or smth!!!
        index = -1
        for i in range(0, len(list_lines), 2):
            line = list_lines[i]
            # For now replace double slash. This should not occur normally.
            if line.decode("utf-8").replace('//', '/') == path:
                index = i
                break
        
        if index == -1:
            raise Exception('Offset not found')

        # ME and secondary string are identified by the same index so we divide by 2
        index = int(index / 2)

        # Read 4 bytes (32bit int) that is the offset in a ROOT file
        offset = int.from_bytes(offsets_buf[index:index + 4], byteorder='little')
        # 57491238
        offset = 57491238
        
        print('Offset: %s' % offset)

        root_obj = self.read_root_object(filename, '', offset)

        # ===================== Connect to rendering process ===================== #

        self.wd = tempfile.mkdtemp()

        print(self.wd)

        self.renderprocess = subprocess.Popen(
            f"dqmRender --state-directory {self.wd}/ > {self.wd}/render.log 2>&1", 
            shell=True, stdout=subprocess.PIPE)
        
        time.sleep(2)

        self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.client.connect(f"{self.wd}/socket")
        
        # , streamerfile = filename.encode("utf-8")
        png, error = self.renderhisto(root_obj, [], name = path, spec='', width=width, height=height, efficiency = bool(False))

        print(error)

        return png        


    def renderhisto(self, th1, refth1s, name = "", spec="", efficiency=False, width=600, height=400, streamerfile=b''):
        DQM_PROP_TYPE_SCALAR = 0x0000000f;
        flags = DQM_PROP_TYPE_SCALAR + 1 # real type is not needed.
        if efficiency:
            flags |= 0x00200000 # SUMMARY_PROP_EFFICIENCY_PLOT
        data = b''
        for o in [th1] + refth1s:
            if isinstance(o, bytes):
                buf = o
            else:
                buf = tobuffer(o)
            data += struct.pack("=i", len(buf)) + buf
        numobjs = len(refth1s) + 1
        nameb = name.encode("utf-8")
        return self.renderbasic(width, height, flags, numobjs, nameb, spec, data, streamerfile)
    
    def renderbasic(self, width, height, flags = 0, numobjs = 1, name = b'', spec = '', data = b'', streamerfile = b''):
        mtype = 4 # DQM_MSG_GET_IMAGE_DATA
        # flags
        vlow = 0
        vhigh = 0
        # numobjs
        # name
        filelen = len(streamerfile)
        namelen = len(name)
        sep = ';' if spec else ''
        specb = f"h={height:d};w={width:d}{sep}{spec}".encode("utf-8")
        speclen = len(specb)
        # data
        datalen = len(data)
        qlen = 0
        msg = struct.pack("=iiiiiiiiii", mtype, flags, vlow, vhigh, numobjs, filelen, namelen, speclen, datalen, qlen)
        msg += streamerfile + name + specb + data
        msg = struct.pack('=i', len(msg) + 4) + msg
        try:
            self.client.send(msg)
            lenbuf = self.client.recv(8)
            errorcode, length = struct.unpack("=ii", lenbuf)
            buf = b''
            while length > 0:
                recvd = self.client.recv(length)
                length -= len(recvd)
                buf += recvd
            return buf, errorcode
        except BrokenPipeError:
            # looks like our renderer died.
            self.dead = True
            return b'', -1



    