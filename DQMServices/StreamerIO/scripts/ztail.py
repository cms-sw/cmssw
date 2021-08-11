#!/usr/bin/env python3

import time
import os
import sys
import zlib
import struct

DECOMPRESS_BUF_SIZE = 4*1024*1024

BLOCK_MAGIC = "\x00\x00\xFF\xFF"
ENDLINE_MAGIC = "\n"

# we treat everything as a deflate stream
# gzip header has no power here
def strip_gzip_header(body):
    assert body[0:2] == "\x1f\x8b"
    method, flags, mtime = struct.unpack("<BBIxx", body[2:10])

    FHCRC    = 0x02
    FEXTRA   = 0x04
    FNAME    = 0x08
    FCOMMENT = 0x10

    i = 10

    if flags & FEXTRA:
        size, = struct.unpack("<H", body[i:i+2])
        i += size + 2

    def skip_until_zero(ix):
        while body[ix] != '\x00': ix += 1
        return ix + 1 
    
    if flags & FNAME:    i = skip_until_zero(i)
    if flags & FCOMMENT: i = skip_until_zero(i)
    if flags & FHCRC:    i += 2

    body = body[i:]
    return body

class Decoder(object):
    def __init__(self, fname, last_n_lines):
        self.f = open(fname, "rb")
        self.last_n_lines = last_n_lines
        self.reset()

    def reset(self):
        self.sync = False
        if hasattr(self, 'zstream'):
            self.zstream.flush()

        self.zstream = zlib.decompressobj(-zlib.MAX_WBITS)

    def decode(self, bytes, if_start=False):
        if not bytes:
            return ""

        if if_start:
            self.sync = True
            #self.zstream = zlib.decompressobj(zlib.MAX_WBITS | 32)
            bytes = strip_gzip_header(bytes)
        elif not self.sync:
            x = bytes.find(BLOCK_MAGIC)
            if x != -1:
                bytes = bytes[x + len(BLOCK_MAGIC):]
                self.sync = True

        if not self.sync:
            # not in sync, can't decode
            return ""
        
        text = self.zstream.decompress(bytes)
        #print "decoded:", len(text), len(self.zstream.unused_data)
        if len(self.zstream.unused_data) == 8:
            # this usually means checksum and len is left
            # but we don't care about any of those!
            self.zstream.flush()
            self.zstream = None

        return text

    def output_line(self, line):
        sys.stdout.write(line)
        sys.stdout.flush()

    def initial_synchronize(self):
        f = self.f

        f.seek(0, 2)
        end = f.tell()
        start = max(0, end - DECOMPRESS_BUF_SIZE)
        f.seek(start, 0)

        body = f.read(end - start)
        text = self.decode(body, start == 0)
        
        self.known_size = end
        return text

    def initial(self):
        text = self.initial_synchronize()

        n_lines = self.last_n_lines
        lines = text.rsplit(ENDLINE_MAGIC, n_lines + 1)
        if len(lines) > n_lines:
            lines = lines[1:]

        self.output_line(ENDLINE_MAGIC.join(lines))

    def follow(self):
        if self.known_size is None:
            raise Exception("Call initial() first.")

        while self.zstream:
            size = os.fstat(self.f.fileno()).st_size

            if self.known_size > size:
                sys.stderr.write("%s: file truncated\n" % sys.argv[0])
                sys.stderr.write("%s: waiting for the next write\n" % sys.argv[0])
                sys.stderr.flush()

                if self.sync:
                    self.sync = False
                    self.zstream.flush()
                    self.zstream = zlib.decompressobj(-zlib.MAX_WBITS)

                text = self.initial_synchronize()
                continue
            elif self.known_size == size:
                time.sleep(1)
                continue

            assert self.f.tell() == self.known_size
            body = self.f.read(size - self.known_size)
        
            text = self.decode(body, self.known_size == 0)
            self.output_line(text)

            self.known_size = size

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='tail, but for gzip files (with Z_FULL_SYNC)')
    parser.add_argument('-f', action="store_true", help='watch the file for changes')
    parser.add_argument('-n', type=int, help='output the last K lines', metavar='K', default=10)
    parser.add_argument('file', help="file name to watch")

    args = parser.parse_args()

    d = Decoder(args.file, args.n)
    d.initial()

    if args.f:
        d.follow()
