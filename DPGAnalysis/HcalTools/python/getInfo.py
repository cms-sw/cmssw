#!/usr/bin/python3
import cx_Oracle
import argparse
import pprint
import json
import os

RUNMIN = os.getenv("HCALDQM_RUNMIN")
conn = os.getenv("HCALDQM_DBCONNECT")
level1= ['HCAL_LEVEL_1']
names = ['HCAL_HBHE', 'HCAL_HF', 'HCAL_HO',  'HCAL_LEVEL_1', 'HCAL_HBHE904', 'HCAL_HF904']

ngrbxidmap="/nfshome0/akhukhun/hcaldqm/config/id2sn_rmcu.json"
ngccmidmap="/nfshome0/akhukhun/hcaldqm/config/id2sn_ngccm.json"
ngqieidmap="/nfshome0/akhukhun/hcaldqm/config/id2sn_qie.json"

def listRuns(runmin, runmax):
    n=5
    db = cx_Oracle.connect(conn)
    cursor = db.cursor()
    p=dict();
    sql="select runnumber, string_value, name from  runsession_parameter where runnumber > {} and runnumber < {} and (".format(runmin, runmax)
    sql = sql + " or ".join(["name=:key"+str(i) for i in range(n*len(level1))])
    sql = sql + ") order by runnumber" 
    for i in range(len(level1)): 
        p["key"+str(n*i)]='CMS.'+level1[i]+":HCAL_TIME_OF_FM_START"
        p["key"+str(n*i+1)]='CMS.'+level1[i]+":FM_FULLPATH"
        p["key"+str(n*i+2)]='CMS.'+level1[i]+":LOCAL_RUNKEY_SELECTED"
        p["key"+str(n*i+3)]='CMS.'+level1[i]+":LOCAL_RUN_KEY" #for older runs
        p["key"+str(n*i+4)]='CMS.'+level1[i]+":EVENTS_TAKEN"
    cursor = cursor.execute(sql, p)
    out={}
    maxfm=0
    maxn=0
    for row in cursor: 
        k = row[2]
        n = row[1]
        r = row[0]
        if r not in out: out[r] = dict(time="", nevents=-1, fm="", key="")
        if(k.endswith("HCAL_TIME_OF_FM_START")): out[r]["time"]=n
        elif(k.endswith("FM_FULLPATH")): 
            fm=n.split("/")[-1]
            out[r]["fm"]=fm
            if(len(fm)>maxfm): maxfm=len(fm)
        elif(k.endswith("EVENTS_TAKEN")): 
            out[r]["nevents"] = int(n)
            if(len(n)>maxn): maxn = len(n)
        else: out[r]["key"]=n
    form="%s | %-24s | %{}d | %-{}s | %s".format(maxn, maxfm)
    for r,i in sorted(out.items()): print (form % (r, i["time"], i["nevents"], i["fm"], i["key"]))
    cursor.close()
    db.close()

def dumpAvailableKeys(run):
    db = cx_Oracle.connect(conn)
    cursor = db.cursor()
    sql="select name from runsession_parameter where runnumber=:run"
    p=dict(run=run)
    cursor = cursor.execute(sql, p)
    res = set();
    for row in cursor: 
        res.add(row[0]);
    cursor.close()
    db.close()
    for i in sorted(res): print(i)

def read(run, key):
    db = cx_Oracle.connect(conn)
    cursor = db.cursor()
    p=dict(run=run)
    OR= " or ".join(["name=:key"+str(i) for i in range(len(names))])
    sql= "".join(["select value from runsession_string where runsession_parameter_id=any(select id from runsession_parameter where (runnumber=:run and (", OR, " )))"])
    for i in range(len(names)): p["key"+str(i)]='CMS.'+names[i]+":"+key
    cursor = cursor.execute(sql, p)
    result = (cursor.fetchone()[0]).read()
    cursor.close();
    db.close()
    return result

def dumpDate(run):
    date = read(run, "HCAL_TIME_OF_FM_START")
    v = date.split()[0].split("-") 
    m = int(v[1])
    d = int(v[2])
    month = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    print("%s%02d" % (month[m-1], d))

def dumpInfo(run):
    db = cx_Oracle.connect(conn)
    cursor = db.cursor()
    p=dict(run=run)
    OR= " or ".join(["name like :key"+str(i) for i in range(len(names))])
    sql= "".join(["select name, string_value from runsession_parameter where runnumber=:run and (", OR, " )"])
    for i in range(len(names)): p["key"+str(i)]='CMS.'+names[i]+":"+"ngRBXManager_Firmware_UniqueIDs_H%"
    cursor = cursor.execute(sql, p)

    row = cursor.fetchone()
    out = dict()
    while row:
        key="{0} ".format(str(row[0]).split("_")[-1])
        result = str(row[1])
        array = result.split(";")
        out[key]=array
        row = cursor.fetchone()
    for key in sorted(out):
        print ("-----------------------\n{0} ".format(key))
        for l in out[key]: print (l)

    cursor.close();
    db.close()

def dumpIDs(run, isqie):
    db = cx_Oracle.connect(conn)
    cursor = db.cursor()
    p=dict(run=run)
    OR= " OR ".join(["name like :key"+str(i) for i in range(len(names))])
    sql= "".join(["select name, string_value from runsession_parameter where runnumber=:run and (", OR, " )"])
    for i in range(len(names)): p["key"+str(i)]='CMS.'+names[i]+":"+"ngRBXManager_Firmware_UniqueIDs_H%"
    cursor = cursor.execute(sql, p)

    row = cursor.fetchone()
    out = dict()
    while row:
        rbx = "{0} ".format(str(row[0]).split("_")[-1])
        n = 2
        if 'M' in rbx: n = 3;
        if 'P' in rbx: n = 3;
        key = "{:s}{:02d}".format(rbx[:n], int(rbx[n:])) 
        
        result = str(row[1])
        array = result.split(";")
        value=[]
        for line in array: 
            rl={}
            w = line.split()
            n = len(w);
            if isqie and w[0]=="QCard": 
                rl['id0'] = w[6]
                rl['id'] = w[7][:-2]
                rl['rm'] = int(w[3][-1])
                rl['qie'] = int(w[1])
            if isqie and w[0]=="CU": 
                rl['id0'] = w[3] 
                rl['id'] = w[4][:-2] 
                rl['rm'] = 5
                rl['qie'] = 1
            if not isqie and (w[0].startswith("ngCCMa") or w[0].startswith("ngCCMb")) and w[2]=="UID:": 
                rl['id0'] = (w[11][1:], "-1")["ERROR" in line]
                rl['id'] = (w[12][:-1], "-1")["ERROR" in line]
                rl['rm'] = 0
                rl['qie'] = 0
            if(rl): value.append(rl)
        if key.startswith("HB") or key.startswith("HE"):
            out[key] = {"value":value, "rbx":rbx}
        row = cursor.fetchone()


    #load id2sn
    data={}
    json_data=open(ngqieidmap).read()
    data.update(json.loads(json_data))

    for key in sorted(out):
        for q in out[key]["value"]:
            id0=q['id0']
            if out[key]["rbx"].startswith('HE'):
                nid0=int(id0, 16);
                id0='0x'+'%08x' % nid0

            k='_'.join((id0, q['id']))
            v=-1
            if k in data: v=data[k]
            print ("%-5s %d %d %-11s %-10s %6d" % (out[key]["rbx"], q['rm'], q['qie'], id0, q['id'], v))

    cursor.close();
    db.close()

def dumpSNs(run):
    db = cx_Oracle.connect(conn)
    cursor = db.cursor()
    p=dict(run=run)
    OR= " OR ".join(["name like :key"+str(i) for i in range(len(names))])
    sql= "".join(["select name, string_value from runsession_parameter where runnumber=:run and (", OR, " )"])
    for i in range(len(names)): p["key"+str(i)]='CMS.'+names[i]+":"+"ngRBXManager_Firmware_UniqueIDs_HB%"
    cursor = cursor.execute(sql, p)

    row = cursor.fetchone()
    out = dict()
    hasRMinfo=False
    hasCCMinfo=False
    while row:
        rbx = "{0} ".format(str(row[0]).split("_")[-1])
        n = 2;
        if 'M' in rbx: n = 3;
        if 'P' in rbx: n = 3;
        key = "{:s}{:02d}".format(rbx[:n], int(rbx[n:])) 

        result = str(row[1])
        array = result.split(";")
        valueCCM=""
        value=""
        i=0
        for line in array: 
            w = line.split()
            n = len(w);
            if w[0]=="QCard": 
                value += "{0}{1}".format(w[7][:-2], ("", " ")[i==3])
                if(i==3): i=0
                else: i=i+1
                hasRMinfo=True
            if w[0]=="CU": value += " {0}".format(w[4][:-2]) 
            if w[0].startswith("ngCCM") and w[2]=="UID:": 
                valueCCM += " {0}0{1} ".format(w[11][1:], w[12][2:-1]) 
                hasCCMinfo=True
        out[key] = {"value":value+valueCCM, "rbx":rbx}
        row = cursor.fetchone()

    data={}
    for f in (ngrbxidmap, ngccmidmap): 
        json_data=open(f).read()
        data.update(json.loads(json_data))
    if(hasRMinfo and hasCCMinfo): print ("RBX   |  RM1  RM2  RM3  RM4   CU CCMa CCMb\n------|-----------------------------------")
    elif hasRMinfo: print ("RBX   |  RM1  RM2  RM3  RM4   CU\n------|-------------------------") 
    else: print("Cannot find published information on RM IDs."); return 
    for key in sorted(out):
        res = ""
        for k in out[key]["value"].split():
            if data.has_key(k): res = res + "{:4d} ".format(data[k])
            elif "NAC" in k: res = res + "   0 "
            else: res = res + "  -1 "
        print ("%-5s | %s" % (out[key]["rbx"].strip(), res))

    cursor.close();
    db.close()

        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="With no arguments, return list of runs with a small summary after runmin={}".format(RUNMIN))
    parser.add_argument("--run", help="run number", type=int);
    parser.add_argument("--runmin", help="earliest run number", type=int, default=RUNMIN);
    parser.add_argument("--runmax", help="earliest run number", type=int, default=99999999999999999);
    parser.add_argument("--key", help="get value for a given key; special values are: sn, qie, ccm, date");
    a=parser.parse_args();
    if(a.run):
        if(a.key=="qie"): dumpIDs(a.run, True)
        elif(a.key=="ccm"): dumpIDs(a.run, False)
        elif(a.key=="raw"): dumpInfo(a.run)
        elif(a.key=="sn"): dumpSNs(a.run)
        elif(a.key=="date"): dumpDate(a.run)
        elif(a.key): print(read(a.run, a.key))
        else: dumpAvailableKeys(a.run)
    else:
        listRuns(a.runmin, a.runmax)
