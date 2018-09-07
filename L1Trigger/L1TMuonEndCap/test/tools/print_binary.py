#!/usr/bin/env python

from __future__ import print_function
import sys

def main():
  if len(sys.argv) < 2:
    print("Usage: %s decimal" % sys.argv[0])
    return

  decimal = int(sys.argv[1])
  print("{0:b}".format(decimal))
  return


# ______________________________________________________________________________
if __name__ == '__main__':

  main()
