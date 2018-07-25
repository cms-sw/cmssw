#!/usr/bin/env python

# External libraries (standard in Python >= 2.4, at least)
from __future__ import print_function
from sys import stdin
from re import split, sub

# Skip the first two lines (headers)
next(stdin)
next(stdin)

print("<MuonAlignment>")
print("")

for line in stdin:
    line = sub("[ \t\n]+$", "", line)
    Alignable, struct1, struct2, struct3, struct4, struct5, \
               relativeto, x, y, z, angletype, angle1, angle2, angle3, \
               xx, xy, xz, yy, yz, zz = split("[ \t]*,[ \t]", line)
    
    print("<operation>")
    if Alignable[0:2] == "DT":
      print(("  <%s " % Alignable), end=' ')
      if struct1 != "":
        print(("wheel=\"%s\" " % struct1), end=' ')
      if struct2 != "":
        print(("station=\"%s\" " % struct2), end=' ')
      if struct3 != "":
        print(("sector=\"%s\" " % struct3), end=' ')
      if struct4 != "":
        print(("superlayer=\"%s\" " % struct4), end=' ')
      if struct5 != "":
        print(("layer=\"%s\" " % struct5), end=' ')
      print("/>")

    if Alignable[0:3] == "CSC":
      print(("  <%s " % Alignable), end=' ')
      if struct1 != "":
        print(("endcap=\"%s\" " % struct1), end=' ')
      if struct2 != "":
        print(("station=\"%s\" " % struct2), end=' ')
      if struct3 != "":
        print(("ring=\"%s\" " % struct3), end=' ')
      if struct4 != "":
        print(("chamber=\"%s\" " % struct4), end=' ')
      if struct5 != "":
        print(("layer=\"%s\" " % struct5), end=' ')
      print("/>")

    if angletype == "phixyz":
      print("  <setposition relativeto=\"%s\" x=\"%s\" y=\"%s\" z=\"%s\" phix=\"%s\" phiy=\"%s\" phiz=\"%s\" />" \
            % (relativeto, x, y, z, angle1, angle2, angle3))
    else:
      print("  <setposition relativeto=\"%s\" x=\"%s\" y=\"%s\" z=\"%s\" alpha=\"%s\" beta=\"%s\" gamma=\"%s\" />" \
            % (relativeto, x, y, z, angle1, angle2, angle3))
      
    print("  <setape xx=\"%s\" xy=\"%s\" xz=\"%s\" yy=\"%s\" yz=\"%s\" zz=\"%s\" />" \
          % (xx, xy, xz, yy, yz, zz))

    print("</operation>")
    print("")

print("</MuonAlignment>")

