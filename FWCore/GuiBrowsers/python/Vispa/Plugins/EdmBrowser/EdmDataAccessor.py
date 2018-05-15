import logging
import os.path

from PyQt4.QtGui import QColor

from Vispa.Share.BasicDataAccessor import BasicDataAccessor
from Vispa.Share.RelativeDataAccessor import RelativeDataAccessor
from Vispa.Share.ParticleDataAccessor import ParticleDataAccessor
from Vispa.Plugins.EventBrowser.EventFileAccessor import EventFileAccessor
from Vispa.Main.Exceptions import exception_traceback
from Vispa.Plugins.EdmBrowser.ParticleDataList import defaultParticleDataList

def eq(self,other):
    return id(self)==id(other)
def ne(self,other):
    return id(self)!=id(other)
# PhysicsTools.PythonAnalysis.cmstools
def all(container):
  # loop over ROOT::TTree and similar
  if hasattr(container,'GetEntries'):
    try:
      entries = container.GetEntries()
      for entry in xrange(entries):
        yield entry
    except:
        raise cmserror("Looping of %s failed" %container) 
  # loop over std::vectors and similar
  elif hasattr(container, 'size'):
    # convert std::map to std::vector<std::pair>
    if hasattr(container, 'ids'):
      container = container.ids()
    try:
      entries = container.size()
      for entry in xrange(entries):
        yield container[entry]
    except:
      pass
  # loop over c buffer
  #elif hasattr(container,'begin') and hasattr(container,'end'):
  #    begin=container.begin()
  #    end=container.end()
  #    while (begin!=end):
  #        yield begin.__deref__()
  #        begin.__preinc__()

class BranchDummy(object):
    def __init__(self,branchtuple):
        self.branchtuple=branchtuple

class EdmDataAccessor(BasicDataAccessor, RelativeDataAccessor, ParticleDataAccessor, EventFileAccessor):
 
    def __init__(self):
        logging.debug(__name__ + ": __init__")

        self._dataObjects = []
        self._edmLabel={}
        self._edmParent={}
        self._edmChildren={}
        self._edmMotherRelations={}
        self._edmDaughterRelations={}
        self._edmChildrenObjects={}
        
        self._eventIndex = 0
        self._numEvents = 0

        self._filename=""
        self._branches=[]
        self._filteredBranches=[]
        self._events=None
        self._readOnDemand=True
        self._underscore=False
        self._filterBranches=True
        self.maxLevels=2
        self.maxDaughters=1000
        
    def isRead(self,object,levels=1):
        if not id(object) in self._edmChildrenObjects.keys():
            return False
        if levels>1 and id(object) in self._edmChildren.keys():
            for child in self._edmChildren[id(object)]:
                if not self.isRead(child, levels-1):
                    return False
        return True

    def children(self,object):
        """ Get children of an object """
        if id(object) in self._edmChildren.keys() and self.isRead(object):
            return self._edmChildren[id(object)]
        else:
            return ()

    def isContainer(self,object):
        """ Get children of an object """
        if id(object) in self._edmChildren.keys() and self.isRead(object):
            return len(self._edmChildren[id(object)])>0
        else:
            return True

    def motherRelations(self,object):
        """ Get motherRelations of an object """
        if id(object) in self._edmMotherRelations.keys():
            return self._edmMotherRelations[id(object)]
        else:
            return ()

    def daughterRelations(self,object):
        """ Get daughterRelations of an object """
        if id(object) in self._edmDaughterRelations.keys():
            return self._edmDaughterRelations[id(object)]
        else:
            return ()

    def label(self,object):
        return self.getShortLabel(object)

    def getShortLabel(self,object):
        if id(object) in self._edmLabel.keys():
            splitlabel=self._edmLabel[id(object)].strip(".").split(".")
            return splitlabel[len(splitlabel)-1]
        else:
            return ""

    def getShortLabelWithType(self,object):
        return self.getShortLabel(object)+" <"+self.getShortType(object)+">"

    def getObjectLabel(self,object):
        splitlabel=self._edmLabel[id(object)].strip(".").split(".")
        return ".".join(splitlabel[1:-1])

    def getType(self,object):
        typ=str(object.__class__)
        if "\'" in typ:
            typ=typ.split("\'")[1]
        if "." in typ:
            typ=typ.split(".")[len(typ.split("."))-1]
        return typ.strip(" ")

    def getShortType(self,object):
        typ=self.getType(object).split("<")[0].strip(" ")
        return typ
    
    def getBranch(self,object):
        entry=object
        while id(entry) in self._edmParent.keys() and self._edmParent[id(entry)]!=None:
            entry=self._edmParent[id(entry)]
        return entry

    def getDepth(self,object):
        entry=object
        i=0
        while id(entry) in self._edmParent.keys() and self._edmParent[id(entry)]!=None:
            entry=self._edmParent[id(entry)]
            i+=1
        return i

    def getObjectProperties(self,object):
        """ get all method properties of an object """
        objects=[]
        for attr in dir(object):
            prop=getattr(object,attr)
            if not attr.startswith("__") and (self._underscore or attr.strip("_")==attr):
                objects+=[(attr,prop)]
        return objects
    
    def getObjectRef(self,object):
        """ get object and resolve references """
        typshort=self.getShortType(object)
        ref_types=["edm::Ref","edm::RefProd","edm::RefToBase","edm::RefToBaseProd","edm::Ptr"]
        value=object
        ref=False
        if typshort in ref_types:
            try:
                if hasattr(object, "isNull") and object.isNull():
                    value="ERROR: "+self.getType(object)+" object is null"
                elif hasattr(object, "isAvailable") and not object.isAvailable():
                    value="ERROR: "+self.getType(object)+" object is not available"
                else:    
                    value=object.get()
                    if isinstance(value, type(None)):
                        value="ERROR: Could not get "+self.getType(object)
                    else:
                        ref=True
            except Exception as message:
                value="ERROR: "+str(message)
        return value,ref

    def getObjectContent(self,object):
        """ get string value of a method """
        if not callable(object):
            return object
        else:
            typ=""
            if not object.__doc__ or str(object.__doc__)=="":
                return "ERROR: Empty __doc__ string"
            docs=str(object.__doc__).split("\n")
            for doc in docs:
                parameters=[]
                for p in doc[doc.find("(")+1:doc.find(")")].split(","):
                    if p!="" and not "=" in p:
                        parameters+=[p]
                if len(parameters)!=0:
                    continue
                typestring=doc[:doc.find("(")]
                split_typestring=typestring.split(" ")
                templates=0
                end_typestring=0
                for i in reversed(range(len(split_typestring))):
                    templates+=split_typestring[i].count("<")
                    templates-=split_typestring[i].count(">")
                    if templates==0:
                        end_typestring=i
                        break
                typ=" ".join(split_typestring[:end_typestring])
            hidden_types=["iterator","Iterator"]
            root_types=["ROOT::"]
            if typ=="" or "void" in typ or True in [t in typ for t in hidden_types]:
                return None
            from ROOT import TClass
            if True in [t in typ for t in root_types] and TClass.GetClass(typ)==None:
                return "ERROR: Cannot display object of type "+typ
            try:
                object=object()
                value=object
            except Exception as message:
                value="ERROR: "+str(message)
            if "Buffer" in str(type(value)):
                return "ERROR: Cannot display object of type "+typ
            else:
                return value

    def isVectorObject(self,object):
        typ=self.getShortType(object)
        return typ=="list" or typ[-6:].lower()=="vector" or typ[-3:].lower()=="map" or typ[-10:].lower()=="collection" or hasattr(object,"size")

    def compareObjects(self,a,b):
        same=False
        if hasattr(a,"px") and hasattr(a,"py") and hasattr(a,"pz") and hasattr(a,"energy") and \
           hasattr(b,"px") and hasattr(b,"py") and hasattr(b,"pz") and hasattr(b,"energy"):
            same=a.px()==b.px() and a.py()==b.py() and a.pz()==b.pz() and a.energy()==b.energy()
        return same

    def getDaughterObjects(self,object):
        """ get list of daughter objects from properties """
        objects=[]
        # subobjects
        objectdict={}
        hidden_attr=["front","back","IsA","clone","masterClone","masterClonePtr","mother","motherRef","motherPtr","daughter","daughterRef","daughterPtr","is_back_safe"]
        broken_attr=[]#["jtaRef"]
        for attr1,property1 in self.getObjectProperties(object):
            if attr1 in hidden_attr:
                pass
            elif attr1 in broken_attr:
                objectdict[attr1]=("ERROR: Cannot read property",False)
            else:
                (value,ref)=self.getObjectRef(self.getObjectContent(property1))
                if not isinstance(value,type(None)) and (not self.isVectorObject(object) or self._propertyType(value)!=None):
                    objectdict[attr1]=(value,ref)
        for name in sorted(objectdict.keys()):
            objects+=[(name,objectdict[name][0],objectdict[name][1],self._propertyType(objectdict[name][0]))]
        # entries in vector
        if self.isVectorObject(object):
            n=0
            for o in all(object):
                (value,ref)=self.getObjectRef(o)
                typ=self._propertyType(value)
                if typ!=None:
                    name="["+str(n)+"]"
                elif "GenParticle" in str(value):
                    name=defaultParticleDataList.getNameFromId(value.pdgId())
                else:
                    name=self.getType(value)+" ["+str(n)+"]"
                objects+=[(name,value,ref,typ)]
                n+=1
        # read candidate relations
        for name,mother,ref,propertyType in objects:
            if hasattr(mother,"numberOfDaughters") and hasattr(mother,"daughter"):
                try:
                    for n in range(mother.numberOfDaughters()):
                        daughter=mother.daughter(n)
                        found=False
                        for na,da,re,st in objects:
                            if self.compareObjects(daughter,da):
                                daughter=da
                                found=True
                        if not id(mother) in self._edmDaughterRelations.keys():
                            self._edmDaughterRelations[id(mother)]=[]
                        self._edmDaughterRelations[id(mother)]+=[daughter]
                        if not id(daughter) in self._edmMotherRelations.keys():
                            self._edmMotherRelations[id(daughter)]=[]
                        self._edmMotherRelations[id(daughter)]+=[mother]
                except Exception as message:
                    logging.error("Cannot read candidate relations: "+str(message))
        return objects

    def _propertyType(self,value):
        if type(value) in (bool,):
            return "Boolean"
        elif type(value) in (int, long):
            return "Integer"
        elif type(value) in (float,):
            return "Double"
        elif type(value) in (complex,str,unicode):
            return "String"
        else:
            return None

    def properties(self,object):
        """ Make list of all properties """
        logging.debug(__name__ + ": properties: "+self.label(object))
        properties=[]

        objectproperties={}
        objectproperties_sorted=[]
        if id(object) in self._edmChildrenObjects.keys():
            for name,value,ref,propertyType in self._edmChildrenObjects[id(object)]:
                if propertyType!=None:
                    objectproperties[name]=(value,propertyType)
                    objectproperties_sorted+=[name]

        properties+=[("Category","Object info","")]
        shortlabel=self.getShortLabel(object)
        properties+=[("String","label",shortlabel)]
        properties+=[("String","type",self.getType(object))]
        objectlabel=self.getObjectLabel(object)
        if objectlabel!="":
            properties+=[("String","object",objectlabel)]
        branchlabel=self.label(self.getBranch(object))
        if shortlabel.strip(".")!=branchlabel.strip("."):
            properties+=[("String","branch",branchlabel)]
        else:
            properties+=[("Category","Branch info","")]
            properties+=[("String","Type",branchlabel.split("_")[0])]
            properties+=[("String","Label",branchlabel.split("_")[1])]
            properties+=[("String","Product",branchlabel.split("_")[2])]
            properties+=[("String","Process",branchlabel.split("_")[3])]

        for property in ["pdgId","charge","status"]:
            if property in objectproperties.keys():
                properties+=[(objectproperties[property][1],property,objectproperties[property][0])]
                del objectproperties[property]

        if "px" in objectproperties.keys():
            properties+=[("Category","Vector","")]
            for property in ["energy","px","py","pz","mass","pt","eta","phi","p","theta","y","rapidity","et","mt","mtSqr","massSqr"]:
                if property in objectproperties.keys():
                    properties+=[(objectproperties[property][1],property,objectproperties[property][0])]
                    del objectproperties[property]

        if "x" in objectproperties.keys():
            properties+=[("Category","Vector","")]
            for property in ["x","y","z"]:
                if property in objectproperties.keys():
                    properties+=[(objectproperties[property][1],property,objectproperties[property][0])]
                    del objectproperties[property]

        if False in [str(value[0]).startswith("ERROR") for value in objectproperties.values()]:
            properties+=[("Category","Values","")]
            for property in objectproperties_sorted:
                if property in objectproperties.keys():
                    if not str(objectproperties[property][0]).startswith("ERROR"):
                        properties+=[(objectproperties[property][1],property,objectproperties[property][0])]
                        del objectproperties[property]
            
        if len(objectproperties.keys())>0:
            properties+=[("Category","Errors","")]
            for property in objectproperties_sorted:
                if property in objectproperties.keys():
                    properties+=[(objectproperties[property][1],property,objectproperties[property][0])]
                
        return tuple(properties)

    def readObjectsRecursive(self,mother,label,edmobject,levels=1):
        """ read edm objects recursive """
        logging.debug(__name__ + ": readObjectsRecursive (levels="+str(levels)+"): "+label)
        # save object information
        if not id(edmobject) in self._edmLabel.keys():
            if not isinstance(edmobject,(int,float,long,complex,str,unicode,bool)):
                # override comparison operator of object
                try:
                    type(edmobject).__eq__=eq
                    type(edmobject).__ne__=ne
                except:
                    pass
            self._edmLabel[id(edmobject)]=label
            self._edmParent[id(edmobject)]=mother
            self._edmChildren[id(edmobject)]=[]
            if not id(mother) in self._edmChildren.keys():
                self._edmChildren[id(mother)]=[]
            self._edmChildren[id(mother)]+=[edmobject]
        if levels==0:
            # do not read more daughters
            return [edmobject],True
        else:
            # read daughters
            return self.readDaughtersRecursive(edmobject,[edmobject],levels)

    def readDaughtersRecursive(self,edmobject,objects,levels=1):
        """ read daughter objects of an edmobject """
        logging.debug(__name__ + ": readDaughtersRecursive (levels="+str(levels)+"): "+str(edmobject))
        # read children information
        if not id(edmobject) in self._edmChildrenObjects.keys():
            self._edmChildrenObjects[id(edmobject)]=self.getDaughterObjects(edmobject)
        # analyze children information
        ok=True
        daughters=self._edmChildrenObjects[id(edmobject)]
        i=0
        for name,daughter,ref,propertyType in daughters:
            # create children objects
            if propertyType==None:
                if ref:
                    label="* "+name
                else:
                    label=name
                if id(edmobject) in self._edmLabel.keys() and self._edmLabel[id(edmobject)]!="":
                    label=self._edmLabel[id(edmobject)]+"."+label
                (res,ok)=self.readObjectsRecursive(edmobject,label,daughter,levels-1)
                objects+=res
            i+=1
            if i>self.maxDaughters:
                logging.warning("Did not read all daughter objects. Maximum is set to "+str(self.maxDaughters)+".")
                return objects,False
        return objects,ok

    def read(self,object,levels=1):
        """ reads contents of a branch """
        logging.debug(__name__ + ": read")
        if isinstance(object,BranchDummy):
            if hasattr(object,"product"):
                return object.product
            if not self._events:
                return object
            try:
                self._events.getByLabel(object.branchtuple[2],object.branchtuple[3],object.branchtuple[4],object.branchtuple[1])
                if object.branchtuple[1].isValid():
                    product=object.branchtuple[1].product()
                    if not isinstance(product,(int,float,long,complex,str,unicode,bool)):
                        # override comparison operator of object
                        try:
                            type(product).__eq__=eq
                            type(product).__ne__=ne
                        except:
                            pass
                    self._dataObjects.insert(self._dataObjects.index(object),product)
                    self._dataObjects.remove(object)
                    self._edmLabel[id(product)]=object.branchtuple[0]
                    object.product=product
                    object=product
                else:
                    self._edmChildrenObjects[id(object)]=[("ERROR","ERROR: Branch is not valid.",False,True)]
                    logging.info("Branch is not valid: "+object.branchtuple[0]+".")
                    object.invalid=True
                    return object
            except Exception as e:
                self._edmChildrenObjects[id(object)]=[("ERROR","ERROR: Unable to read branch : "+str(e),False,True)]
                object.unreadable=True
                logging.warning("Unable to read branch "+object.branchtuple[0]+" : "+exception_traceback())
                return object
        if self.isRead(object,levels):
            return object
        if levels>0:
            self.readDaughtersRecursive(object,[],levels)
        return object

    def goto(self, index):
        """ Goto event number index in file.
        """
        self._eventIndex=index-1
        self._edmLabel={}
        self._edmChildren={}
        self._edmMotherRelations={}
        self._edmDaughterRelations={}
        self._edmChildrenObjects={}
        if self._events:
            self._events.to(self._eventIndex)
        self._dataObjects=[]
        i=0
        for branchtuple in self._filteredBranches:
            branch=BranchDummy(branchtuple)
            self._dataObjects+=[branch]
            self._edmLabel[id(branch)]=branchtuple[0]
            if not self._readOnDemand:
                self.read(branch,self.maxLevels)
            i+=1
        if self._filterBranches and self._events:
            self.setFilterBranches(True)
        return True
    
    def eventNumber(self):
        return self._eventIndex+1

    def numberOfEvents(self):
        return self._numEvents

    def topLevelObjects(self):
        return self._dataObjects

    def open(self, filename=None):
        """ Open edm file and show first event """
        self._filename=filename
        self._branches=[]
        if os.path.splitext(filename)[1].lower()==".txt":
            file = open(filename)
            for line in file.readlines():
                if "\"" in line:
                    linecontent=[l.strip(" \n").rstrip(".") for l in line.split("\"")]
                    self._branches+=[(linecontent[0]+"_"+linecontent[1]+"_"+linecontent[3]+"_"+linecontent[5],None,linecontent[1],linecontent[3],linecontent[5])]
                else:
                    linecontent=line.strip("\n").split(" ")[0].split("_")
                    if len(linecontent)>3:
                        self._branches+=[(linecontent[0]+"_"+linecontent[1]+"_"+linecontent[2]+"_"+linecontent[3],None,linecontent[1],linecontent[2],linecontent[3])]
        elif os.path.splitext(filename)[1].lower()==".root":
            from DataFormats.FWLite import Events, Handle
            self._events = Events(self._filename)
            self._numEvents=self._events.size()
            branches=self._events.object().getBranchDescriptions()
            for branch in branches:
                try:
                    branchname=branch.friendlyClassName()+"_"+branch.moduleLabel()+"_"+branch.productInstanceName()+"_"+branch.processName()
                    handle=Handle(branch.fullClassName())
                    self._branches+=[(branchname,handle,branch.moduleLabel(),branch.productInstanceName(),branch.processName())]
                except Exception as e:
                    logging.warning("Cannot read branch "+branchname+":"+str(e))
        self._branches.sort(lambda x, y: cmp(x[0], y[0]))
        self._filteredBranches=self._branches[:]
        return self.goto(1)

    def particleId(self, object):
        charge=self.property(object,"pdgId")
        if charge==None:
            charge=0
        return charge
        
    def isQuark(self, object):
        particleId = self.particleId(object)
        if not particleId:
            return False
        return defaultParticleDataList.isQuarkId(particleId)

    def isLepton(self, object):
        particleId = self.particleId(object)
        if not particleId:
            return False
        return defaultParticleDataList.isLeptonId(particleId)

    def isGluon(self, object):
        particleId = self.particleId(object)
        if not particleId:
            return False
        return defaultParticleDataList.isGluonId(particleId)

    def isBoson(self, object):
        particleId = self.particleId(object)
        if not particleId:
            return False
        return defaultParticleDataList.isBosonId(particleId)

    def isPhoton(self, object):
        particleId = self.particleId(object)
        if not particleId:
            return False
        if not hasattr(defaultParticleDataList,"isPhotonId"):
            return False
        return defaultParticleDataList.isPhotonId(particleId)
    
    def isHiggs(self, object):
        particleId = self.particleId(object)
        if not particleId:
            return False
        if not hasattr(defaultParticleDataList,"isHiggsId"):
            return False
        return defaultParticleDataList.isHiggsId(particleId)
    
    def lineStyle(self, object):
        particleId = self.particleId(object)
        if hasattr(defaultParticleDataList,"isPhotonId") and defaultParticleDataList.isPhotonId(particleId):
            return self.LINE_STYLE_WAVE
        elif defaultParticleDataList.isGluonId(particleId):
            return self.LINE_STYLE_SPIRAL
        elif defaultParticleDataList.isBosonId(particleId):
            return self.LINE_STYLE_DASH
        return self.LINE_STYLE_SOLID
    
    def color(self, object):
        particleId = self.particleId(object)
        if defaultParticleDataList.isLeptonId(particleId):
            return QColor(244, 164, 96)
        elif defaultParticleDataList.isQuarkId(particleId):
            return QColor(0, 100, 0)
        elif hasattr(defaultParticleDataList,"isHiggsId") and defaultParticleDataList.isHiggsId(particleId):
            return QColor(247, 77, 251)
        elif defaultParticleDataList.isBosonId(particleId):
            return QColor(253, 74, 74)
        return QColor(176, 179, 177)
    
    def charge(self, object):
        charge=self.property(object,"charge")
        if charge==None:
            charge=0
        return charge
    
    def linkMother(self, object, mother):
        pass
        
    def linkDaughter(self, object, daughter):
        pass

    def underscoreProperties(self):
        return self._underscore
    
    def setUnderscoreProperties(self,check):
        self._underscore=check
    
    def filterBranches(self):
        return self._filterBranches
    
    def setFilterBranches(self,check):
        if not self._events:
            return True
        self._filterBranches=check
        if check:
            for branch in self._dataObjects[:]:
                result=self.read(branch,0)
                if isinstance(result,BranchDummy):
                    self._dataObjects.remove(result)
                if hasattr(result,"invalid"):
                    self._filteredBranches.remove(result.branchtuple)
            return True
        else:
            self._filteredBranches=self._branches[:]
            self.goto(self.eventNumber())
            return False

    def filteredBranches(self):
        return self._filteredBranches
    