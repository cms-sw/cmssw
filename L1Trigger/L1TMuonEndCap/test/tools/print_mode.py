#!/usr/bin/env python

import sys

def main():
  if len(sys.argv) < 2:
    print("Usage: %s mode" % sys.argv[0])
    return

  mode = int(sys.argv[1])

  if mode > 15:
    raise Exception("mode must be 0-15")

  printme = []
  if (mode & 8):  printme.append("1")
  if (mode & 4):  printme.append("2")
  if (mode & 2):  printme.append("3")
  if (mode & 1):  printme.append("4")
  printme = "-".join(printme)

  print("{0:s}".format(printme))
  return


# ______________________________________________________________________________
if __name__ == '__main__':

  main()
