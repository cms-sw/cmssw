#!/usr/bin/env python

import sys

def main():
  if len(sys.argv) < 3:
    print("Usage: %s straightness layer_code" % sys.argv[0])
    return

  straightness = int(sys.argv[1])
  layer_code = int(sys.argv[1])

  if straightness > 7:
    raise Exception("straightness must be 0-7")
  if layer_code > 7:
    raise Exception("layer_code must be 0-7")

  quality_code = 0
  quality_code = ( 
    (((straightness>>2) & 1) << 5) |
    (((straightness>>1) & 1) << 3) |
    (((straightness>>0) & 1) << 1) |
    (((layer_code>>2)   & 1) << 4) |
    (((layer_code>>1)   & 1) << 2) |
    (((layer_code>>0)   & 1) << 0)
  )

  print("0b{0:b}".format(quality_code))
  return


# ______________________________________________________________________________
if __name__ == '__main__':

  main()
