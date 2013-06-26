#
# File: validate-o2o-wbm.py (Zongru Wan)
#

f1 = open("o2o.log", "r")
oTSC = ""
oGT = ""
oGTRUNSETTINGS = ""
while True:
    a = f1.readline()
    if a[0] == '-':
        break
    if a.find("Current TSC key = ") != -1:
        oTSC = a[18:].strip()
    elif a.find("GT ") != -1:
        oGT = a[3:].strip()
    elif a.find("L1GtPrescaleFactorsAlgoTrigRcd ") != -1:
        oGTRUNSETTINGS = a[31:].strip()
f1.close()

f2 = open("wbm.log", "r")
wTSC = f2.readline().strip()
wGT = f2.readline().strip()
wGTRUNSETTINGS = f2.readline().strip()
f2.close()

f = open("val.log", "w")
if oTSC == wTSC and oGT == wGT and oGTRUNSETTINGS == wGTRUNSETTINGS:
    f.write("successful")
else:
    f.write("failed")
f.close()
