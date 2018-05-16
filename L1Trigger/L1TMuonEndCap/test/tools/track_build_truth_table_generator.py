# Generate 2^6 = 64 entries
# a=12, b=23, c=34, d=13, e=14, f=24

entries = []
for a in [1,0]:
  for b in [1,0]:
    for c in [1,0]:
      for d in [1,0]:
        for e in [1,0]:
          for f in [1,0]:
            entry = (a,b,c,d,e,f)
            entries.append(entry)

## Debug
#for entry in entries:
#  print entry

# ______________________________________________________________________________
# Encode the truth table logic
#
# From Andrew Brinkerhoff (2017-11-30):
#   * All stations which are kept in the track must have a valid dTheta (< max) to all other stations kept in the track.
#   * 3-station tracks are always preferred over 2-station tracks.
#   * Tracks where the first station is ME1 are most-preferred, followed by ME2, followed by ME3.
#   * After that, tracks where the second station is ME2 are most-preferred, followed by ME3, followed by ME4.

results = []
for entry in entries:
  (a,b,c,d,e,f) = entry

  # Everyone is good
  if a and b and c and d and e and f:
    vmask = (1,1,1,1)

  # 12, 23, 13 are good
  elif a and b and d:
    vmask = (1,1,1,0)  # mask s4

  # 12, 24, 14 are good
  elif a and f and e:
    vmask = (1,1,0,1)  # mask s3

  # 13, 34, 14 are good
  elif d and c and e:
    vmask = (1,0,1,1)  # mask s2

  # 23, 34, 24 are good
  elif b and c and f:
    vmask = (0,1,1,1)  # mask s1

  # 12 is good
  elif a:
    vmask = (1,1,0,0)  # mask s3 and s4

  # 13 is good
  elif d:
    vmask = (1,0,1,0)  # mask s2 and s4

  # 14 is good
  elif e:
    vmask = (1,0,0,1)  # mask s2 and s3

  # 23 is good
  elif b:
    vmask = (0,1,1,0)  # mask s1 and s4

  # 24 is good
  elif f:
    vmask = (0,1,0,1)  # mask s1 and s3

  # 34 is good
  elif c:
    vmask = (0,0,1,1)  # mask s1 and s2

  # else
  else:
    vmask = (0,0,0,0)  # fail

  results.append(vmask)


# ______________________________________________________________________________
# Output
for result in results:
  (w,x,y,z) = result
  print "%i%i%i%i" % (z,y,x,w)

# Output as C++ array
generate_cpp_array = False
if generate_cpp_array:
  linebreak = "\n    "
  sep = ", "
  s = "static const int trk_bld[64] = {"
  s += linebreak
  i = 0
  for result in results:
    (w,x,y,z) = result
    s += "0b%i%i%i%i" % (z,y,x,w)
    if i != 63: s += sep
    if i%8 == 7 and i != 63:  s += linebreak
    i += 1
  s += "\n};"
  print s
