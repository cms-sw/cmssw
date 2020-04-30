import re
import time
import zlib
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
        # TODO: offline_data should probably be removed. No GUI flavor distinction is required.
        samples = {'samples': [{
            'type': 'offline_data',
            'items': []
        }]}

        results = []
        if run:
            run = '%%%s%%' % run
        if dataset:
            dataset = '%%%s%%' % dataset

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

        for row in results:
            samples['samples'][0]['items'].append({
                'run': row[0],
                'dataset': row[1]
            })

        return samples

    def get_archive(self, run, dataset, path, search):

        def get_subitem(full_path, current_dir):
            """Returns a closest sub item of full_path inside current_dir
            The second value of the response tuple indicates if that closest sub item is final one
            i.e. if it is a file or directory.
            If full_path is a/b/c/d/file and current_dir is /a/b/c function will return (d, False).
            If full_path is a/b/c/d/file and current_dir is /a/b/c/d function will return (file, True).
            If current_dir is not part of full_path, function will return False.
            """

            # Remove double slashes
            full_path = full_path.replace('//', '/')
            current_dir = current_dir.replace('//', '/')

            if full_path.startswith(current_dir):
                names = full_path.replace(current_dir, '').split('/')
                if len(names) == 1: # This is an ME
                    return (names[0], True)
                else: # This is a folder
                    return (names[0], False)
            else:
                return False

        # Path must end with a slash
        if path and not path.endswith('/'):
            path = path + '/'

        blob = self.get_me_list_blob(run, dataset)

        # Uncompress and get the ME directory structure (flattened list of MEs).
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
        buf = zlib.decompress(blob)
        lines = buf.split(b'\n')

        regex = re.compile(search) if search else None
        dirs = set()
        objs = set()

        # When doing dir listing we don't care about the secondary string
        # TODO: This is now a linear search over a sorted list: optimize this!!!
        for x in range(0, len(lines), 2):
            line = lines[x].decode("utf-8")
            subitem = get_subitem(line, path)
            if subitem:
                if regex and not regex.match(line.split('/')[-1]):
                    continue # Regex is provided and ME name doesn't match it

                if subitem[1]:
                    objs.add(subitem[0])
                else:
                    dirs.add(subitem[0])

        # Remove last slash before returning
        path = path[:-1]

        # Put results to a list to be returned
        data = {'contents': []}
        data['contents'].extend({'subdir': x} for x in dirs)
        data['contents'].extend({'obj': x, 'dir': path} for x in objs)

        return data

    def get_me_list_blob(self, run, dataset):
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



    def read_root_object(self, filename, me_path, offset):
        """Returns binary representation of ROOT object wether it's a histogram or string ME.
        No file handle is passed here to make sure that the return value of this method could be cached."""

        filename = filename.replace('root://eoscms.cern.ch/', '')
        file = uproot.open(filename)
        cur = uproot.source.cursor.Cursor(offset)
        key = uproot.rootio.TKey.read(file.source, cur, file._context, None)
        cur = key._cursor.copied()
        buf = bytes(cur.bytes(key._source, key._fObjlen))

        if key._fClassName == b"TObjString":
            # In the case of a string, name contains the value in XML-like format.
            return key._fName
        else:
            # tobject, reconstruct frame for TBufferFile
            # The format is <@length><kNewClassTag=0xFFFFFFFF><classname><nul><@length><2 bytes version><data ...
            # @length is 4byte length of the *entire* remaining object with bit 0x40 (kByteCountMask)
            # set in the first (most significant) byte. This prints as "@" in the dump...
            # the data inside the TKey seems to have the version already.
            classname = key._fClassName
            totlen = 4 + len(classname) + 1 + len(buf)
            head = struct.pack(">II", totlen | 0x40000000, 0xFFFFFFFF)
            return head + classname + b'\0' + buf