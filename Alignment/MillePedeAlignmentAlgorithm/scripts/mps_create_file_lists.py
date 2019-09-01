#!/usr/bin/env python

from __future__ import print_function

print("=======================================")
print("This script is moved to its new home in")
print("tkal_create_file_lists.py              ")
print("(in Alignment/CommonAlignment/scripts).")
print("=======================================")
print()
import subprocess
execfile(subprocess.check_output(["which", "tkal_create_file_lists.py"]).rstrip())
