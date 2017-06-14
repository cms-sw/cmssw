#!/usr/bin/python

import Image, sys

background = Image.open(sys.argv[1])
foreground = Image.open(sys.argv[2])

background.paste(foreground, (3000, 40), foreground)
background.save("result.png")
