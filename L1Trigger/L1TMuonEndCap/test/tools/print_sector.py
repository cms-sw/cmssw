#!/usr/bin/env python

import sys
from math import degrees, pi

def main():
  if len(sys.argv) < 2:
    print("Usage: %s radian" % sys.argv[0])
    return

  rad = eval(sys.argv[1])
  if rad <= 0.:
    rad += 2*pi
  deg = degrees(rad)
  print("rad: {0} deg: {1}".format(rad, deg))
  
  sector = int((deg - 15)/60.) + 1
  print("{0:d}".format(sector))
  return


# ______________________________________________________________________________
if __name__ == '__main__':

  main()
