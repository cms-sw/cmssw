"""A script to check the GE0 geometry with the ME0 geometry. They
should have the same position, pitch, etc. with the only difference
being the detids (ME0 uses ME0DetId, GE0 uses GEMDetId).

I.J. Watson (ian.james.watson@cern.ch)
"""

import re
import numpy as np

fge0 = open('GE0testOutput.out')
fme0 = open('ME0testOutput.out')

# parse the files
ge0 = {}
me0 = {}

cstripper = lambda s: s[:s.find(',')]
# GEMEtaPartition 1, GEMDetId = 688406528,  Re -1 Ri 1 St 0 La 1 Ch 1 Ro 1
fl = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?") # (0=before exp, -1=after exp)
findfls = lambda l: [float(f[0]+f[-1]) for f in fl.findall(l)]
crnt = None
top = re.compile("top(x,y,z)[cm] = (134.714, -2.4311e-13, 527), top (eta,phi) = (2.07314, 0)")
for line in fge0:
    words = line.split()
    if words[0] == "GEMEtaPartition":
        detid = np.array(words)[6::2] # the detid int's are different, so use the translation
        if crnt: ge0[crnt[0]] = crnt[1]
        crnt = (tuple(detid),{})
    elif crnt:
        fls = findfls(line)
        if words[0] == "nStrips": # nStrips = 384, nPads =  192
            crnt[1]["nStrips"] = int(cstripper(words[2]))
            crnt[1]["nPads"] = int(words[5])
        elif words[0] == "Dimensions[cm]:": # Dimensions[cm]: b = 42.302, B = 47.5074, H  = 14.7606
            crnt[1]["b"], crnt[1]["B"], crnt[1]["H"] = fls
        elif words[0] == "top(x,y,z)[cm]":
            crnt[1]["tx"], crnt[1]["ty"], crnt[1]["tz"], crnt[1]["te"], crnt[1]["tp"],  = fls
        elif words[0] == "center(x,y,z)[cm]":
            crnt[1]["cx"], crnt[1]["cy"], crnt[1]["cz"], crnt[1]["ce"], crnt[1]["cp"],  = fls
        elif words[0] == "bottom(x,y,z)[cm]":
            crnt[1]["bx"], crnt[1]["by"], crnt[1]["bz"], crnt[1]["be"], crnt[1]["bp"],  = fls
        elif words[0] == "pitch": # pitch (top,center,bottom) = (0.123717, 0.116939, 0.110161), dEta = 0.112809, dPhi = 20
            crnt[1]["px"], crnt[1]["py"], crnt[1]["pz"], crnt[1]["pe"], crnt[1]["pp"],  = fls
            

ge0[crnt[0]] = crnt[1]

# ME0EtaPartition 1 , ME0DetId = 704651526,  Region 1 Layer 1 Chamber 1 EtaPartition 1 
crnt = None
for line in fme0:
    words = line.split()
    if words[0] == "ME0EtaPartition":
        re, la, ch, ro = np.array(words)[7::2] # the detid int's are different, so use the translation
        detid = re, '1', '0', la, ch, ro
        if crnt: me0[crnt[0]] = crnt[1]
        crnt = (tuple(detid),{})
    elif crnt:
        # crnt[1].append(line)
        # nStrips = 384, nPads =  192
        fls = findfls(line)
        if words[0] == "nStrips":
            crnt[1]["nStrips"] = int(words[2][:words[2].find(',')])
            crnt[1]["nPads"] = int(words[5])
        elif words[0] == "Dimensions[cm]:": # Dimensions[cm]: b = 42.302, B = 47.5074, H  = 14.7606
            crnt[1]["b"], crnt[1]["B"], crnt[1]["H"] = fls
        elif words[0] == "top(x,y,z)[cm]": #top(x,y,z)[cm] = (149.524, 1.75305e-13, -527), top(eta,phi) = (-1.97243, 0)
            crnt[1]["tx"], crnt[1]["ty"], crnt[1]["tz"], crnt[1]["te"], crnt[1]["tp"],  = fls
        elif words[0] == "center(x,y,z)[cm]":
            crnt[1]["cx"], crnt[1]["cy"], crnt[1]["cz"], crnt[1]["ce"], crnt[1]["cp"],  = fls
        elif words[0] == "bottom(x,y,z)[cm]":
            crnt[1]["bx"], crnt[1]["by"], crnt[1]["bz"], crnt[1]["be"], crnt[1]["bp"],  = fls
        elif words[0] == "pitch": # pitch (top,center,bottom) = (0.123717, 0.116939, 0.110161), dEta = 0.112809, dPhi = 20
            crnt[1]["px"], crnt[1]["py"], crnt[1]["pz"], crnt[1]["pe"], crnt[1]["pp"],  = fls

me0[crnt[0]] = crnt[1]

for k in me0.keys():
    if k not in ge0.keys():
        print("Missing key:", k, "is in ME0 but not in GE0")
        break
else:
    print("No missing keys in GE0")
for k in ge0.keys():
    if 0 != k[2]: continue # check station
    if k not in me0.keys():
        print("Missing key:", k, "is in GE0 but not in ME0")
        break
else:
    print("No missing keys in ME0")

for k, v in me0.items():
    for kk, vv in v.items():
        if ge0[k][kk] != vv:
            print("bad ", kk, k,v, ge0[k])
            break
    else: continue
    break
else:
    print("No differences between ME0 and GE0 in checked settings")
