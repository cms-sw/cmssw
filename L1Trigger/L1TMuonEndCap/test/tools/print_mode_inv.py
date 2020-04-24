#!/usr/bin/env python

import sys

def main():
  if len(sys.argv) < 2:
    print("Usage: %s mode_inv" % sys.argv[0])
    return

  mode_inv = int(sys.argv[1])

  if mode_inv > 15:
    raise Exception("mode_inv must be 0-15")

  printme = []
  if (mode_inv & 1):  printme.append("1")
  if (mode_inv & 2):  printme.append("2")
  if (mode_inv & 4):  printme.append("3")
  if (mode_inv & 8):  printme.append("4")
  printme = "-".join(printme)

  print("{0:s}".format(printme))
  return


# ______________________________________________________________________________
if __name__ == '__main__':

  main()
