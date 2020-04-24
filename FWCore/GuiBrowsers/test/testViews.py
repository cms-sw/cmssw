#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

if __name__ == "__main__":
    tests = []
    dirs = [os.path.abspath(os.path.join(os.path.dirname(__file__),"Views"))]
    print "scanning directories:"
    for dir in dirs:
        print os.path.basename(dir)
        sys.path.append(dir)
        tests += [f[: - 3] for f in os.listdir(dir) if f.startswith("test") and f.endswith(".py")]
    print "---"
    print "running tests:"
    for test in sorted(tests):
        print test
    print "---"
    for test in sorted(tests):
        exec("from " + str(test) + " import *")
    unittest.main()
