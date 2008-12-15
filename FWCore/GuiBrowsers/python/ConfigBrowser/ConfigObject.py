import os.path

# CMSSW imports
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Modules as mod

from RelativeObject import *

class ConfigObject(RelativeObject):
    """ Object that holds cms object information """
    file_dict={}
    def __init__(self,object=None,mother=None,type="default"):
        """ constructor """
        RelativeObject.__init__(self,mother,type)
        self.object=object
        self.inputtags=self.getInputTags()

    def getLabel(self):
        """ Get label of this """
        text=""
        if hasattr(self.object,"label_"):
            text=self.object.label_()
        if text=="":
            if hasattr(self.object,"type_"):
                text=self.object.type_()
        return text

    def setLabel(self,label):
        raise NotImplementedError
            
    label=property(getLabel,setLabel)

    def getType(self):
        """ Get type of this """
        text=""
        if isinstance(self.object,cms.Path):
            text="path"
        elif isinstance(self.object,cms.Sequence):
            text="sequence"
        elif isinstance(self.object,cms.Source):
            text="source"
        elif isinstance(self.object,mod._Module):
            text="module"
        elif isinstance(self.object,cms.Service):
            text="service"
        elif isinstance(self.object,cms.ESSource):
            text="es_source"
        elif isinstance(self.object,cms.ESProducer):
            text="es_producer"
        elif isinstance(self.object,cms.ESPrefer):
            text="es_prefer"
        elif isinstance(self.object,cms.PSet):
            text="pset"
        return text

    def getClassname(self):
        """ Get classname of this """
        text=""
        if hasattr(self.object,"type_"):
            text=self.object.type_()
        return text

    def getFullFilename(self):
        """ Get full filename """
        text=""
#        if hasattr(self.object,"_filename"):
#            text=self.object._filename
        if text=="" or text.find("FWCore/ParameterSet")>=0 or text.find("/build/")>=0:
            if self.label in self.file_dict:
                text=self.file_dict[self.label]
        root=os.path.splitext(text)[0]
        if root!="":
            text=root+".py"
        return text

    def getLineNumber(self):
        """ Get linenumber """
        text=""
        if hasattr(self.object,"_filename"):
            if self.object._filename.find("FWCore/ParameterSet")<0 and self.object._filename.find("ConfigBrowser")<0:
                if hasattr(self.object,"_lineNumber"):
                    text=str(self.object._lineNumber)
        return text

    def getFilename(self):
        """ Get filename """
        text=os.path.splitext(os.path.basename(self.getFullFilename()))[0]
        return text

    def getPackage(self):
        """ Get Package of this file """
        shortdirname=os.path.dirname(self.getFullFilename()).split('python/')
        text=""
        if len(shortdirname)>1:
            text=shortdirname[1]
        return text

    def getParameters(self):
        """ Get parameters of this """
        this_parameters=[]
        if hasattr(self.object,"parameters_"):
            this_parameters=self.object.parameters_().items()
        elif hasattr(self.object,"_seq"):
            this_parameters=[('sequence',self.object._seq.dumpSequencePython())]
        if hasattr(self.object,"targetLabel_"):
            this_parameters+=[('targetLabel',self.object.targetLabel_())]
        return this_parameters

    def addInputTag(self,value,this_key,this_inputtags):
        """ Add alls inputtags of value to a list """
        if isinstance(value,list):
            for i in value:
                self.addInputTag(i,this_key,this_inputtags)
        if hasattr(value,"parameters_"):
            this_inputtags+=self.readInputTagsRecursive(value.parameters_().items(),this_key)
        if isinstance(value,cms.InputTag):
            pythonValue=value.pythonValue()
            this_inputtags+=[(str(this_key),value.pythonValue().split(",")[0].strip("\""))]

    def readInputTagsRecursive(self,this_parameters,start_key=""):
        """ Make list of inputtags from parameter dict """
        this_inputtags=[]
        for key,value in this_parameters:
            this_key=start_key
            if this_key!="":
                this_key+="."
            this_key+=key
            self.addInputTag(value,this_key,this_inputtags)
        return this_inputtags

    def getInputTags(self):
        """ Make list of inputtags from parameter dict """
        return self.readInputTagsRecursive(self.getParameters())

    def getTopEntry(self):
        entry=self
        while entry.mothers!=[]:
            entry=entry.mothers[0]
        return entry

    def getFoundInRecursive(self,entry,foundin):
        """ Find mother sequences recursively """
        for daughter in entry.daughters:
            if self.label==daughter.label and not entry.label in foundin:
                foundin+=[entry.label]
        for daughter in entry.daughters:
            self.getFoundInRecursive(daughter,foundin)
        return foundin
        
    def getFoundIn(self):
        """ Make list of all mother sequences """
        return self.getFoundInRecursive(self.getTopEntry(),[])

    def getUses(self):
        """ Get list of all configobjects that are used as input """
        uses=[]
        for key,value in self.inputtags:
            if value not in uses:
                uses+=[value]
        return uses
    
    def getUsedByRecursive(self,entry,usedby):
        """ Find configobjects that use this as input recursively """
        if hasattr(entry,"getUses"):
            for uses in entry.getUses():
                if self.label==uses and not entry.label in usedby:
                    usedby+=[entry.label]
        for daughter in entry.daughters:
            self.getUsedByRecursive(daughter,usedby)
        return usedby
        
    def getUsedBy(self):
        """ Find configobjects that use this as input """
        return self.getUsedByRecursive(self.getTopEntry(),[])
        
    def getProperties(self):
        """ Make list of all properties """
        properties=[]
        properties+=[("Label","Object info","")]
        if self.label!="":
            properties+=[("Text","label",self.label)]
        if self.getType()!="":
            text=self.getType()
            if self.getClassname()!="":
                text+=" <"+self.getClassname()+">"
            properties+=[("Text","type",text)]
        if self.getFilename()!="":
            text=self.getFilename()
            if self.getLineNumber()!="":
                text+=" : "+self.getLineNumber()
            properties+=[("Text","file",text)]
        if self.getPackage()!="":
            properties+=[("Text","package",self.getPackage())]
        if self.getFullFilename()!="":
            properties+=[("Text","full filename",self.getFullFilename())]
        foundin=self.getFoundIn()
        if len(foundin)>0:
            text=""
            for entry in foundin:
                if text!="":
                    text+=", "
                text+=entry
            properties+=[("Text","in sequence",text)]
        if len(self.getUses())+len(self.getUsedBy())>0:
            properties+=[("Label","Connections","")]
            if len(self.getUses())>0:
                text=""
                for label in self.getUses():
                    if text!="":
                        text+=", "
                    text+=label
                properties+=[("Text","uses",text)]
            if len(self.getUsedBy())>0:
                text=""
                usedby=[]
                for entry in self.getUsedBy():
                    if text!="":
                        text+=", "
                    text+=entry
                properties+=[("Text","used by",text)]
        if len(self.getParameters())>0:
            properties+=[("Label","Parameters","")]
            for key,value in sorted(self.getParameters()):
                try:
                    properties+=[("Text",str(key),str(value))]
                except AttributeError:
                    pass
#                    print "problem reading parameters..."
        return properties

    def setProperties(self,properties):
        raise NotImplementedError
            
    properties=property(getProperties,setProperties)

#    def __getattr__(self,name):
#        return getattr(self.object,name)
