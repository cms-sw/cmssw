from  xml.dom.minidom import *
import os
import string
import sys
import glob
import math

# output directory
CONFIGFILE = "configuration.xml"
#CONFIGFILE = "config_test.xml"
outdir = "./data/"
#outdir = "./Test/"
soldir = ""
#soldir = "solids"
GENSOLID_ONCE = {}
LOGFILE = open("log.txt",'w')
LOGFILE.write("Logfile\n")
#delete previously generated tables
cmd = "rm -rf " + outdir
os.system(cmd)
os.mkdir(outdir)
#os.mkdir(outdir+soldir)
#for i in glob.glob(outdir+"*"):
#    try:
#       os.remove(i)
#    except:
#       print "ex"
        

##################################################################################
#  Solid actions
##################################################################################

# generic for atomic solids ######################################################
# input: table-name, dom-element, current-namespace, other #######################
#      postfix = any other text to be written end of the row, must end with a ',
#      prefix  = any other text to be written beginning of a row, must end with a ','
# output: entry in file 'table-name' of all attributes of 'dom-element' ##########
def generic_solid(table,element,ns,postfix="",prefix="",docpost="[postfix]",docpre="[prefix],"):
    if GENSOLID_ONCE.has_key(table) == 0:
        GENSOLID_ONCE[table]=1
        f = open(outdir+soldir+table+".cdl",'w')
        s = "-- "
        if len(prefix):
            s = s + docpre
        for att in element.attributes.values(): 
            s = s + att.name + ','
        if len(postfix):
            s = s + docpost
        f.write(s[:-1]+"\n")
        f.close()
    file = open(outdir+soldir+table,'a')
    s=prefix
    for att in element.attributes.values():
        if att.name == "name":
            s = s + qname(att.value,ns) + ','
        else:
            s = s + unitc(att.value).__str__() + ','
    s = s + postfix 
    file.write(s[:-1]+"\n")

# examines the sensfullness of the pcone/phedra z-sections or rz-points (currently z-section only)
def examine_section(r,ns,kind):
    # currently only z-sections
    if kind != "ZS,":
        return
    sections = r.getElementsByTagName("ZSection")
    z = []
    rmin = []
    rmax = []
    for s in sections:
        z.append(unitc(s.getAttribute("z")))
        rmin.append(unitc(s.getAttribute("rMin")))
        rmax.append(unitc(s.getAttribute("rMax")))
    #print z        
    # check, whether z-values come in order
    xmin = z[0]
    for x in z:
        if x < xmin:
            LOGFILE.write(r.tagName + " " + qname(r.getAttribute("name"),ns) + " z values not z1 <= z2\n")
        xmin = x
    for i in range(0,len(rmin)):
        if rmin[i] == rmax[i]:
            LOGFILE.write(r.tagName + " " + qname(r.getAttribute("name"),ns) + " rmin = rmax\n")
            if (i != 0) and (i != (len(rmin)-1)):
                LOGFILE.write("   the one above might be a problem!\n")
                print r.tagName + " " + qname(r.getAttribute("name"),ns) + " rmin = rmax\n", i, range(0,len(rmin))
                
    for i in range(0,len(rmin)):
        if rmin[i] > rmax[i]:
            LOGFILE.write(r.tagName + " " + qname(r.getAttribute("name"),ns) + " rmin > rmax\n")
    # check pairs of same z-values
    zz = {}
    for i in range(0,len(rmin)):
        zz[z[i]] = 1
        #print z[i], zz
        if i < len(rmin)-1:
            err = 0
            if z[i] == z[i+1]:
                if rmax[i+1] < rmin[i]:
                    err = 1
                if rmin[i+1] > rmax[i]:
                    err = 2
            if err>0:
                LOGFILE.write(r.tagName + " " + qname(r.getAttribute("name"),ns) + " discontinuity at " + i.__str__() + "\n")
    if len(zz) == 2:
         #print zz, qname(r.getAttribute("name"),ns)
         #print n
         LOGFILE.write(r.tagName + " " + qname(r.getAttribute("name"),ns) + " simpler solid possible\n")
        
def poly_section_action(r,ns,kind):
    count = 0
    name = qname(r.getAttribute("name"),ns) + ","
    if r.tagName == "Polycone":
        name = name + ","
    else:
        name = "," + name
    #print kind
    if kind=="ZS,":
        for i in r.getElementsByTagName("ZSection"):
            prefix = count.__str__() + ","
            generic_solid("ZSECTIONS.dat",i,ns,name,prefix,"polycone_solid_name,polyhedra_solid_name,","sequence_no,")
            count = count + 1
    else:
        for i in r.getElementsByTagName("RZPoint"):
            prefix = count.__str__() + ","
            generic_solid("RZPOINTS.dat",i,ns,name,prefix,"polycone_solid_name,polyhedra_solid_name,","sequence_no,")
            count = count + 1
    #if count<5:
    #    LOGFILE.write("ZSECTION:" +" " + r.tagName + " " + name + "\n")
    examine_section(r,ns,kind)
        
def boolean_components(r,ns):
    s = ""
    for c in r.getElementsByTagName("rSolid"):
        s = s + qname(c.getAttribute("name"),ns) + ","
    t = r.getElementsByTagName("Translation")
    if len(t) != 0:
        s = s + unitc(t[0].getAttribute("x")).__str__() + ","
        s = s + unitc(t[0].getAttribute("y")).__str__() + ","
        s = s + unitc(t[0].getAttribute("z")).__str__() + ","
    else:
        s = s + "0,0,0,"    
    r = r.getElementsByTagName("rRotation")
    if len(r) != 0:
        s = s + qname(r[0].getAttribute("name"),ns) + ","
    else:
        s = s + "rotations:UNIT,"
    return s

# Solid-action  ##########################################################################
# called during the SOLIDS-table generation
# each action creates a sub-table containing the data for the solids
# some actions create several sub-tables (e.g. polycones, rz-sectsion ...)

def box_action(r,s,ns):
    s = s + r.tagName +','
    generic_solid("BOXES.dat",r,ns)
    return s

def reflectionsolid_action(r,s,ns):
    s = s + r.tagName +','
    generic_solid("REFLECTIONSOLIDS.dat",r,ns)
    return s
    
def shapelesssolid_action(r,s,ns):
    s = s  + r.tagName +','
    generic_solid("SHAPELESSSOLIDS.dat",r,ns)
    return s

def tubs_action(r,s,ns):
    s = s  + r.tagName +','
    generic_solid("TUBESECTIONS.dat",r,ns)
    return s

def tube_action(r,s,ns):
    s = s + r.tagName +','
    generic_solid("TUBES.dat",r,ns)
    return s

def trd1_action(r,s,ns):
    s = s + r.tagName +','
    generic_solid("TRD1S.dat",r,ns)
    return s

def polyhedra_action(r,s,ns):
    kind = "RZ"
    if len(r.getElementsByTagName("ZSection"))>0:
        kind = "ZS,"
    poly_section_action(r,ns,kind)    
    s = s + r.tagName +','
    generic_solid("POLYHEDRAS.dat",r,ns,kind,"","RZ_or_ZS,")
    return s

def polycone_action(r,s,ns):
    kind = "RZ"
    if len(r.getElementsByTagName("ZSection"))>0:
        kind = "ZS,"
    poly_section_action(r,ns,kind)
    generic_solid("POLYCONES.dat",r,ns,kind,"","RZ_or_ZS,")
    s = s + r.tagName +','   
    return s

def cone_action(r,s,ns):
    s = s + r.tagName +','
    generic_solid("CONES.dat",r,ns)
    return s

def pseudotrap_action(r,s,ns):
    s = s + r.tagName +','
    generic_solid("PSEUDOTRAPEZOIDS.dat",r,ns)
    return s

def trapezoid_action(r,s,ns):
    s = s + r.tagName +','
    generic_solid("TRAPEZOIDS.dat",r,ns)
    return s

def intersectionsolid_action(r,s,ns):
    s = s + r.tagName +','
    comp = boolean_components(r,ns)
    generic_solid("BOOLEANSOLIDS.dat",r,ns,"I,"+comp,"","operation,solidA,solidB,x,y,z,rot")
    return s

def unionsolid_action(r,s,ns):
    s = s + r.tagName +','
    comp = boolean_components(r,ns)
    generic_solid("BOOLDEANSOLIDS.dat",r,ns,"U,"+comp,"","operation,solidA,solidB,x,y,z,rot")
    return s

def subtractionsolid_action(r,s,ns):
    s = s + r.tagName +','
    comp = boolean_components(r,ns)
    generic_solid("BOOLEANSOLIDS.dat",r,ns,"S,"+comp,"","operation,solidA,solidB,x,y,z,rot")
    return s
    
# global things
CATEGORIES = {}
SOLIDTYPES = {
               'SubtractionSolid':subtractionsolid_action,
               'UnionSolid':unionsolid_action,
               'IntersectionSolid':intersectionsolid_action,
               'Trapezoid':trapezoid_action,
               'PseudoTrap':pseudotrap_action,
               'Cone':cone_action,
               'Polycone':polycone_action,
               'Polyhedra':polyhedra_action,
               'Trd1':trd1_action,
               'Tube':tube_action,
               'Tubs':tubs_action,
               'ShapelessSolid':shapelesssolid_action,
               'ReflectionSolid':reflectionsolid_action,
               'Box':box_action,
               }

rad_to_deg = math.pi/180.
UNITS = { u'deg':rad_to_deg,
          u'fm':1e-12,
          u'nm':1e-9,
          u'mum':1e-3,
          u'mm':1.,
          u'cm':10.,
          u'm':1000.,
          u'rad':1,
          u'g/cm3':1,
          u'mg/cm3':0.001,
          u'g/mole':1.,
          u'mg/mole':0.001
        }

PPCOUNT = [ 0 ]

# unitconversion
def unitc(s):
    try:
        s = string.strip(s)
        x = string.split(s,'*')
        if len(x) == 1:
            return s
        if len(x) == 2:
            return string.atof(x[0])*UNITS[x[1]]  
    except:
        print "ERROR IN unitc: >" + s + "<"
        print x
        print UNITS[x[1]]       

def get_docs(config):
    result = []
    print CONFIGFILE
    doc = xml.dom.minidom.parse(open(CONFIGFILE,'r'))
    file_elements = doc.getElementsByTagName("File")
    print `file_elements`   
    for file_element in file_elements:
        file_name = file_element.getAttribute("name")
        url_name = file_element.getAttribute("url")
        print url_name + file_name
        path = url_name+file_name
        ns = string.split(path,'/')[-1][:-4]
        result.append([url_name+file_name,ns])
    return result

##################################################################################
#  other  actions
##################################################################################

def rot_action(r,s,ns):
    #print r.tagName
    if r.tagName=="ReflectionRotation":
        s = s + "1,"
    else:
        s = s + "0,"
    return s

# "ABC:abc" -> "ABC:abc", "abc" -> "ns:abc"
def qname(s,ns):
    if string.find(s,':') == -1:
        s = ns +':' + s
    return s

def comp_action(e,s,ns):
    file = open(outdir+"MATERIALFRACTIONS.dat",'a')
    fracs = e.getElementsByTagName("MaterialFraction")
    for frac in fracs:
        fm = frac.getAttribute("fraction")
        mat = qname(frac.getElementsByTagName("rMaterial")[0].getAttribute("name"),ns)
        result = "" + qname(e.getAttribute("name"),ns) + ',' + mat + ',' + fm + "\n"
        file.write(result)
    return s

def log_action(logp,s,ns):
    # isolate the category, name_cat[0]=name, name_cat[1]=category
    name_cat = string.split(s,',')
    CATEGORIES[name_cat[1]] ='' # put category in global dictionary
    #print name_cat[1]
    nodelist = logp.childNodes
    l = nodelist.length
    mat =""
    sol =""
    for i in range(0,l-1):
        if string.find(nodelist[i].nodeName,"Ma") != -1:
            mat = qname(nodelist[i].getAttribute("name"),ns)
        if string.find(nodelist[i].nodeName,"So") != -1:
            sol = qname(nodelist[i].getAttribute("name"),ns)
    result =  name_cat[0] + ',' + sol + ',' + mat + ',' + name_cat[1] + ',' 
    return result
    
def pos_action(posp,s,ns):
    PPCOUNT[0] = PPCOUNT[0] + 1
    nodelist = posp.childNodes
    l = nodelist.length
    parent = ""
    child = ""
    tra = []
    rot = ""
    for i in range(0,l-1):
        nn = nodelist[i].nodeName
        if string.find(nn,'Pa') != -1:
            parent = qname(nodelist[i].getAttribute("name"),ns)
        if string.find(nn,'Child') != -1:
            child = qname(nodelist[i].getAttribute("name"),ns)
        if string.find(nn,'Tra') != -1:
            tra.append(unitc(nodelist[i].getAttribute("x")))
            tra.append(unitc(nodelist[i].getAttribute("y")))
            tra.append(unitc(nodelist[i].getAttribute("z")))
        if string.find(nn,'Rot') != -1:
            rot = qname(nodelist[i].getAttribute("name"),ns)
    result = PPCOUNT[0].__str__() + ',' + s 
    if len(tra) != 0:
        result = result + tra[0].__str__() + ',' + tra[1].__str__() + ',' + tra[2].__str__() + ','
    else:
        result = result + ",,,"
    result = result + rot + ',' + parent + ',' + child + ','
    return result
    
docs_ns = get_docs("configuration.xml")
print `docs_ns`
for doc in docs_ns:
    document_path = doc[0]
    namespace = doc[1]
    print "Processing document: " + doc[0] + " ns=" + doc[1] 
    doc = xml.dom.minidom.parse(open(doc[0],'r'));

    # syntax: list( tablename, list(elements), list(attributes) )
    # prefixing an attribute name with ":"  =  convert to a q-name
    # prefixing an attribute name with ";"  =  do NOT perform unit-conversion
    table = [
              [ 'MATERIALS.dat', ['ElementaryMaterial', 'CompositeMaterial'] ,
                [':name','density'] ,0 ],

              [ 'ELEMENTARYMATERIALS.dat', ['ElementaryMaterial'],
                [':name', ';atomicNumber', 'atomicWeight',  ';symbol' ], 0 ],

              [ 'COMPOSITEMATERIALS.dat', ['CompositeMaterial'],
                [':name',';method' ], comp_action ],

              [ 'ROTATIONS.dat', ['Rotation', 'ReflectionRotation'],
                [':name', 'thetaX', 'phiX', 'thetaY', 'phiY', 'thetaZ', 'phiZ'], rot_action ],

              [ 'LOGICALPARTS.dat', ['LogicalPart'],
                [':name', ';category', 'itemid' ], log_action ],

              [ 'POSPARTS.dat', ['PosPart'],
                ['copyNumber'], pos_action ],

              [ 'SOLIDS.dat', SOLIDTYPES.keys(),
                [':name'], SOLIDTYPES.values() ]
            
            ]

    for i in table:
        tablename = i[0]
        filename=outdir+tablename
        file = open(outdir+tablename,'a')    
        elements = i[1]
        attributes = i[2]
        action = i[3]
        elCount = -1
        for el in elements:
            elCount = elCount + 1
            els = doc.getElementsByTagName(el)
            for e in els:
                s = ""
                for at in attributes:
                    flagQ = 0 # q-name
                    flagU = 0 # unit converson
                    if at[0]==':': # process a q-name
                        flagQ = 1
                        at = at[1:]
                    if at[0]==';': # don't do a unit-conversion
                        flagU = 1
                        at = at[1:]
                    attr = e.getAttribute(at)    
                    if flagQ == 1: # process a q-name
                        attr = qname(attr,namespace)
                    else:
                        if flagU == 0:    
                             attr = unitc(attr)
                        else:     
                             if attr == " ": # fix symbol=" " in ElementaryMaterial (could be done in an action)
                                 attr = "0"
                    s = s + attr.__str__() + ","
                if action != 0: # trigger special action if defined in the processing-table
                    le=0
                    try:
                        le=len(action)
                    except TypeError:
                        s = action(e,s,namespace)
                    else:
                        s = action[elCount](e,s,namespace)
                #print s[:len(s)-1]
                file.write(s[:len(s)-1]+"\n")
        file.close()

file = open(outdir+"CATEGORIES.dat",'w')
for i in CATEGORIES.keys():
    file.write(i+"\n")
file.close()

LOGFILE.close()
