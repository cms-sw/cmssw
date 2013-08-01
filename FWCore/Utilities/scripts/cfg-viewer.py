#!/usr/bin/env python
# -*- coding: latin-1 -*-
import re
import collections
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.SequenceTypes as seq
## doing types

class unscheduled:
  def __init__(self,cfgFile,html,quiet,helperDir,fullDir):
    self._html = html
    self._quiet = quiet
    self._theDir= fullDir
    self._helperDir = helperDir
    self._mother,self._daughter ={},{}
    self._reg = re.compile("<|>|'")
    self._dictParent,self._modSeqParent = "DictParent","ModuleSeqParent"
    self._prodConsumParent = "ProdConsumParent"
    self._type = "%stypes.js"%(fullDir)
    self._allJSFiles =["types.js"]
    self._data,self._types,self._genericTypes={},{},{}
    self._simpleData,self._complexData="simpleData", "complexData"
    self._prodConsumData = "prodData"
    self._simpleFile, self._complexFile = ["%s%s.js"%(self._theDir, 
                            x) for x in [self._simpleData,self._complexData]] 
    self._prodConsumFile = "%s%s.js"%(self._theDir,self._prodConsumData)
    self._config= ConfigDataAccessor.ConfigDataAccessor()
    self._config.open(cfgFile)
    self._proceed()

  def _proceed(self):
    self._filename= ""
    self._getData(self._config.topLevelObjects())
    ty = "genericTypes"
    with open(self._type, 'w') as f:
      f.write("var %s=%s"%(ty,genericTypes))
    self._createObjects()
    self._writeDictParent(ty)
    self._writeModSeqParent()
    self._writeProdConsum()
    JS = ["%s%s"%(self._helperDir,x)for x in self._allJSFiles]
    html(self._html,JS,self._data, self._theDir, self._helperDir)

  def _getData(self,objs):
    # i will loop around objs and keep adding things which are configFodlers
    calc =[]
    for each in objs:
      name = self._config.label(each)
      kids = self._config.children(each)
      if(not kids):
        if(not self._quiet):print name, "is empty."
        continue
      # Taking liberty of assuming all are of same type.
      ty = type(kids[0])
      if(ty is ConfigDataAccessor.ConfigFolder):
        objs.extend(kids)
      else:
        if(not self._quiet):print "computing %s.."%(name)
        if(isinstance(kids[0], seq._ModuleSequenceType)):
          # e.g.it's a path, sequence or endpath
          self._doSequenceTypes(kids,name)
        else:
          # e.g. it's a pset etc.
          self._doNonSequenceType(kids,name)
        calc.append(name)
    # now i will print out producers/consumers.
    if(self._quiet):print "calculated: %s."%(", ".join(calc))
    self._producersConsumers()

  # Get the data for data which is a SequenceType
  def _doSequenceTypes(self,paths,name):
    theDataFile = self._calcFilenames(name)
    fullDataFile = "%s%s"%(self._theDir,theDataFile)
    topLevel = self._calcFilenames("top-"+name)
    fullTopLevel = "%s%s"%(self._theDir,topLevel)
    json = [topLevel,theDataFile]
    cap = name.capitalize()
    bl={}
    types = False
    foundany=False
    with open(fullDataFile,'w') as data:
      data.write("{")
      v = visitor(data)  
      for value in paths:
        if(not types):
          generic, spec = re.sub("['>]", "", 
                             str(value.__class__)).split(".")[-2:]
          doTypes(spec,generic)
          self._saveData(spec,self._complexData,json) 
          types = True
        # Dont think we need to check for this here.
        mo =self._config.motherRelations(value)
        dau = self._config.daughterRelations(value)
        if(mo or dau and not foundany):
          self._mother[name] = [self._config.label(i) for i in mo]
          self._daughter[name] = [self._config.label(i) for i in dau]
          foundany = True
        key = self._config.label(value)
        value.visit(v)
        bl[key]= getParamSeqDict(v._finalExit(),
                    self._config.filename(value), "")
      data.write("}")
      with open(fullTopLevel, 'w') as other:
        other.write(JSONFormat(bl))

  def _doNonSequenceType(self,items, globalType):
    everything={}
    foundany=False
    types = False
    theDataFile =self._calcFilenames(globalType)
    fullDataFile = "%s%s"%(self._theDir,theDataFile)
    always = False
    if(globalType =="modules"):
      self._saveData(globalType.capitalize(),self._simpleData,[theDataFile]) 
      always = True
      types = True
    for each in items:
      if(always or not types):
        generic, spec = re.sub("['>]", "", 
                          str(each.__class__)).split(".")[-2:]
        doTypes(spec,generic)
        if(not types):
          self._saveData(spec,self._simpleData,[theDataFile]) 
          types = True
      name = self._config.label(each)
      mo =self._config.motherRelations(each)
      dau = self._config.daughterRelations(each)
      if(mo or dau and not foundany):
        foundany = True
        self._mother[name] = [self._config.label(i) for i in mo]
        self._daughter[name] = [self._config.label(i) for i in dau]
      filename = self._config.filename(each)
      theType = self._config.type(each)
      if(isinstance(each,cms._Parameterizable)):
        out = getParameters(each.parameters_())
      elif(isinstance(each,cms._ValidatingListBase)):
        out = listBase(each)
      everything[name] = getParamSeqDict(out, filename, theType)
    with open(fullDataFile,'w') as dataFile:
      dataFile.write(JSONFormat(everything))

  def _calcFilenames(self,name):
    return "data-%s.json"%(name)

  def _producersConsumers(self):
    if(not self._mother and not self._daughter):
      return
    # TODO should really make this dynamic incase name changes.    
    for name,theDict in {"producer":self._mother, 
                         "consumer":self._daughter}.iteritems():
      thedataFile = self._calcFilenames(name)
      fulldataFile = "%s%s"%(self._theDir,thedataFile)
      self._saveData(name.capitalize(),self._prodConsumData,
                      [thedataFile,"data-modules.json"]) 
      #               "%s %s"%(thedataFile,"data-modules.json")) 
      with open(fulldataFile,'w') as moth:
        moth.write(JSONFormat(theDict))

  def _saveData(self,name,base,jsonfiles):
    jsonfiles = " ".join(["%s%s"%(self._helperDir,x)for x in jsonfiles])
    temp={}
    temp["data-base"] = base
    temp["data-files"] = jsonfiles
    self._data[name] = temp

  #TODO make nicer
  def _createObjects(self):
    self._allJSFiles.append("%s.js"%(self._simpleData))
    #self._allJSFiles.append(self._complexFile)
    #self._allJSFiles.append(self._prodConsumFile)
    self._allJSFiles.append("%s.js"%(self._complexData))
    self._allJSFiles.append("%s.js"%(self._prodConsumData))

    base = "obj= Object.create(new %s(%s));"
    format="""
    function %s(%s){
      var obj;
      %s
      return obj;
    }
    """
    name = "data"
    paramName="modules"
    simple = base%(self._dictParent,name)
    with open(self._simpleFile, 'w') as setUp:
      setUp.write(format%(self._simpleData,paramName,self._load(
                                           name,paramName,simple)))
    
    secName = "topL"
    paramName=["modules","topLevel"]
    complexOne = base%(self._modSeqParent,"%s,%s"%(name, secName))
    with open(self._complexFile, 'w') as setUp:
      setUp.write(format%(self._complexData,", ".join(paramName),
                    self._load(name,paramName[0],
                    self._load(secName, paramName[1],complexOne))))

    secName = "topL"
    paramName=["modules","topLevel"]
    complexOne = base%(self._prodConsumParent,"%s,%s"%(name, secName))
    with open(self._prodConsumFile , 'w') as setUp:
      setUp.write(format%(self._prodConsumData,", ".join(paramName),
                    self._load(name,paramName[0],
                    self._load(secName, paramName[1],complexOne))))

  def _load(self,name,param,inner):
    return"""
      loadJSON(%s).done(function(%s){\n%s\n});
    """%(param,name,inner)

  def _writeDictParent(self, typeName):
    exVar ='this.%(key)sKey= "%(key)s";'
    exFunc ="""
/**
 * Gives the %(key)s 
 * @param {String} the key to the dictionary.
 * @returns {String || Integer} result from dictionary if found else 0.
 */
%(name)s.prototype.get%(key)s = function(key){
  return this.getFeature(key,this.%(key)sKey);
}
    """
    extra= """
    /**
    /**
* Uses the list of parents names to go further into
* the lists in the dictionry, and give the last parents children.
* @param {Array} the names of parents to look through.
*                 First parent should be a key to the dictionary. 
* @param {Integer} the index where the elusive parameter is. Needed incase 
*         we have a list with muultiple parameters of the same name.
*                 
* @returns {Array} the resulting children from the last parent.
*/
DictParent.prototype.getInnerParams = function(parents, index){
  var currentList = this.data[parents[0]][this.%(key)sKey];
  var targetList=[]
  var siblingsOfTarget=[]
  for(var i=1; i < parents.length;i++){
    for(var p=0; p < currentList.length;p++){
      if(currentList[p][0]==parents[i]){
        targetList = currentList[p]
        siblingsOfTarget=currentList
        var found = targetList[1];
        break;
      }
    }
    currentList = found;
  }
  if(p != index)targetList=siblingsOfTarget[index]
  return targetList;
}
    """
    functs=""
    variables =""
    for feature in dictFeatures:
      functs +=exFunc%{"key":feature,"name":self._dictParent}
      if(feature == "Parameters"):
        functs +=extra%{"key": feature, "name":self._dictParent}
      variables +=exVar%{"key": feature}
    fileName= "%s%s.js"%(self._theDir, self._dictParent)
    self._allJSFiles.append("%s.js"%(self._dictParent))
    with open(fileName, 'w') as parent:
      parent.write("""
/*
  Base Object for dictionaries.
  To use, create a dictParent obj,
  then use Object.create to create a new obj
  then you can change some variables/add functions
  and have the inherited functions.
*/
function %(name)s(data){
  this.data=data;
  %(theVars)s
}
/**
 * Finds the desired feature from a dictionary.
* @param {String} the key to the dictionary.
* @param {String} the feature we want from the dictionary.
* @returns {String || Integer} result from dictionary if found else 0.
*/
%(name)s.prototype.getFeature = function(key, feature){
  var temp = this.data[key];
  if(temp) return temp[feature];
  else return 0;
}
%(getterFunctions)s

/**
 * Gives the keys from desired dictionary.
* @returns {Array} all keys from the dictionary.
*/
%(name)s.prototype.getKeys = function(){
  return Object.keys(this.data);
}

/**
 * Gives the generic type for a given type.
 * @param {String} type 
 * @returns {String} the generic type
 */
function getGenericType(type){
    return %(gen)s[type];
}
   """%{"theVars":variables,"getterFunctions":functs, 
        "gen": typeName, "name":self._dictParent})

  def _writeModSeqParent(self):
    fileName= "%s%s.js"%(self._theDir,self._modSeqParent)
    self._allJSFiles.append("%s.js"%(self._modSeqParent))
    with open(fileName, 'w') as parent:
      parent.write("""
/* 
 Base object for thing of the type: 
 ._ModuleSequenceType - i.e. paths,endpaths.sequences 
 It also inherits from DictParent.           
*/                                          
function %(name)s(data,topLevel, nameList,indexList){ 
  this.data = data; 
  this.topLevelData=topLevel;// e.g. pathNames to module names 
  this.fixedNameList = nameList; // e.g.names of paths 
  this.modulesToNameIndex = indexList; 
  // e.g. module names and list of numbers
  // corresponding to paths in the namelist.
}
%(name)s.prototype = new %(dict)s(this.data);
/**
 * Gives the direct children
 * @param {String} a path name
 * @returns {Array} list of names of the the children.
 */
%(name)s.prototype.getModules = function(name){ 
  return this.topLevelData[name][this.ParametersKey];
} 
/**
 * Gives all paths a module is a child of.
 * @param {String} a module name 
 * @returns {Array} list of path names.
 */
%(name)s.prototype.getPaths= function(theMod){
  var listOfIndexes = this.modulesToNameIndex[theMod];     
  var resultingPaths =[];
  for (var i=0; i < listOfIndexes.length; i++){
    resultingPaths.push(this.fixedNameList[i])
  } 
  return resultingPaths;        
} 
%(name)s.prototype.getKeys = function(){
  return Object.keys(this.topLevelData)
}
%(name)s.prototype.getTopFile = function(key){
  return this.topLevelData[key][this.FileKey];
}

    """%{"name": self._modSeqParent, "dict": self._dictParent})

  #TODO duplicate code
  def _writeProdConsum(self):
    fileName= "%s%s.js"%(self._theDir,self._prodConsumParent)
    self._allJSFiles.append("%s.js"%(self._prodConsumParent))
    with open(fileName, 'w') as parent:
      parent.write("""
/* 
 Base object for thing of the type: 
 ._ModuleSequenceType - i.e. paths,endpaths.sequences 
 It also inherits from DictParent.           
*/                                          
function %(name)s(data,topLevel, nameList,indexList){ 
  this.data = data; 
  this.topLevelData=topLevel;// e.g. pathNames to module names 
  this.fixedNameList = nameList; // e.g.names of paths 
}
%(name)s.prototype = new %(dict)s(this.data);
/**
 * Gives the direct children
 * @param {String} a path name
 * @returns {Array} list of names of the the children.
 */
%(name)s.prototype.getModules = function(name){ 
  return this.topLevelData[name];
}
%(name)s.prototype.getKeys = function(){
  return Object.keys(this.topLevelData)
}
%(name)s.prototype.getTopFile = function(key){
  return this.getFile(key);
}

    """%{"name": self._prodConsumParent, "dict": self._dictParent})

# Helper function which gets parameter details.
def getParameters(parameters):
  all =[]
  if(not parameters):
    return []
  for (name,valType) in parameters.iteritems():
    theT= (valType.configTypeName() if(
           hasattr(valType,"configTypeName")) else "").split(" ",1)[-1]
    temp = re.sub("<|>|'", "", str(type(valType)))
    generic, spec = temp.split(".")[-2:]
    doTypes(spec,generic)
    theList=[name]
    if(isinstance(valType,cms._Parameterizable)):
      theList.append(getParameters(valType.parameters_()))
    elif(isinstance(valType,cms._ValidatingListBase)):
      theList.append(listBase(valType))
    else:
      if(isinstance(valType,cms._SimpleParameterTypeBase)):
        value = valType.configValue()
      else:
        try:
          value = valType.pythonValue()
        except:
          value = valType
      theList.append(value) 
      if(theT != "double" and theT !="int" and type(valType)!= str):
        if(not valType.isTracked()):
          theList.append("untracked")
    theList.append(theT)
    all.append(theList)
  return all

def listBase(VList):
  # we have a list of things.. 
  #loop around the list get parameters of inners.
  #Since ValidatingListBases "*usually* have the same types 
  #throughout, just test first --is this a rule?
  # Find out and if needed move these ifs to the loop.
  if(not VList):return ""
  first = VList[0]
  if(isinstance(first,cms._Parameterizable)or 
     isinstance(first,cms._ValidatingListBase)):
    anotherVList=False
    if(isinstance(first,cms._ValidatingListBase)):
      anotherVList=True
    outerList=[]
    for member in VList:
      if(member.hasLabel_()):
        name = member.label()
      else:
        name = member.configTypeName()
      innerList=[name]
      if(not anotherVList):
        #print "in listbase"
        innerList.append(getParameters(member.parameters_()))
      else:
        innerList.append(listBase(member))
      temp = re.sub("<|>|'", "", str(type(member)))
      generic, spec = temp.split(".")[-2:]
      doTypes(spec,generic)
      innerList.append(spec)
      outerList.append(innerList)
    return outerList
  elif(isinstance(first,cms._SimpleParameterTypeBase)):
    return ",".join(i.configValue() for i in VList)
  elif(isinstance(first,cms._ParameterTypeBase)):
    return ",".join(i.pythonValue() for i in VList)
  else:
    #Most things should at least be _ParameterTypeBase, adding this jic
    try:
      outcome = ",".join(str(i) for i in VList)
      return outcome
    except:
      return "Unknown types"

  
dictFeatures=["Parameters", "Type", "File"]
# Used to enforce dictionary in datfile have same format.
def getParamSeqDict(params, fil, typ):
  d={}
  d[dictFeatures[0]] = params
  d[dictFeatures[1]] = typ
  d[dictFeatures[2]] = fil
  return d

class visitor:
  def __init__(self, df):
    self._df = df
    self._underPath = [] # direct children of paths 
                         #(includes children of modules)
    self._modulesToPaths={} # map from modules to index of path -
    self._seq = 0
    self._pathLength=0
    self._currentName =""
    self._oldNames =[]
    self._done =[]
    self._seqs={}
    self._typeNumbers = {}
    self._innerSeq = False
    self._reg= re.compile("<|>|'")

  def _finalExit(self):
    self._pathLength+=1
    temp = self._underPath
    self._underPath =[]
    return temp

  # Only keep in if we manage to move doModules into this class
  def _getType(self,val):
    return re.sub(self._reg, "", str(type(val))).split(".")[-2:]


  """
    Do Module Objects e.g. producers etc
  """
  def _doModules(self,modObj, dataFile, seq, seqs, currentName, innerSeq):
    name = modObj.label_()
    # If this is not an inner sequence then we add so it can go to paths
    if(seq==0):
      self._underPath.append(name)
    else:
        seqs[currentName].append([name])
    # If we've seen this name before, no need to compute values again.
    # we need to put this mod/seq name under the path name in the dict
    if(name not in self._modulesToPaths.keys()):
      self._modulesToPaths[name] =[self._pathLength]
      filename = modObj._filename.split("/")[-1]
      generic,specific = self._getType(modObj)
      doTypes(specific,generic)
      d = getParamSeqDict(getParameters(modObj.parameters_()),
                          filename, specific)
      theS='"%s":%s'
      if(len(self._modulesToPaths.keys()) > 1): theS=","+theS
      dataFile.write(theS%(name, JSONFormat(d))) 
    else:
      #oldMods.append(name) 
      self._modulesToPaths[name].append(self._pathLength)

  def enter(self, value):
    if(isinstance(value,cms._Module)):
      self._doModules(value, self._df, self._seq,
                self._seqs, self._currentName, self._innerSeq)
    elif(isinstance(value,cms._Sequenceable)):
      generic,specific = self._getType(value)
      doTypes(specific, generic)
      if(isinstance(value, cms._ModuleSequenceType)):
        if(len(self._currentName) >0):
          self._oldNames.insert(0, self._currentName)
        name = value.label()
        if(self._seq >0):
          # this is an inner sequence
          self._innerSeq = True;
          self._seqs[self._currentName].append([name])
        else:
          self._underPath.append(name)
        self._seqs[name] = []
        self._currentName = name
        self._seq +=1
      else:
        # just sequenceable.. 
        name = value.__str__()
        if(self._currentName):
          self._seqs[self._currentName].append([name, specific])
        else:
          self._underPath.append(value.__str__())
          if(name not in self._done):
            d = getParamSeqDict([], "", specific)
            #print "enter"
            self._df.write(',"%s":%s'%(name,JSONFormat(d))) 
            self._done.append(name) 

  def leave(self, value):
    if(isinstance(value,cms._Module)):
      return
    elif(isinstance(value,cms._Sequenceable)):
      # now need to determine difference between 
      #ones which have lists and ones which dont
      if(isinstance(value, cms._ModuleSequenceType)):
        name = value.label()
        if(name in self._oldNames):self._oldNames.remove(name)
        if(self._currentName == name):
          if(self._oldNames):
            self._currentName = self._oldNames.pop(0)
          else:
            self._currentName=""
        if(name not in self._done):
          generic,specific = self._getType(value)
          d = getParamSeqDict(self._seqs.pop(name), "", specific)
          #print "leave", d
          self._df.write(',"%s":%s'%(name,JSONFormat(d)))
          self._done.append(name)  
        self._seq -=1
        if(self._seq==0): self._innerSeq = False;

class html:
  def __init__(self,name,js,items,theDir, helperDir):
    jqName = "%scfgJS.js"%(theDir)
    jqLocal = "%scfgJS.js"%(helperDir)
    css = "%sstyle.css"%(theDir)
    cssL = "%sstyle.css"%(helperDir)
    js.insert(0,jqLocal)
    self._jqueryFile(jqName)
    self._printHtml(name,self._scripts(js),self._css(css, cssL),
                    self._items(items))

  def _scripts(self, js):
    x = """<script type="text/javascript" src="%s"></script>"""
    return "\n".join([x%(i) for i in js])

  def _items(self, items):
    l = """<input type="radio" name="docType" value="%(n)s"
    data-base="%(d)s"data-files="%(f)s"/>
    <span name="Info" value="%(n)s">%(n)s </span>"""
    s= [l%({"n":x,"f":y["data-files"],"d":y["data-base"]})
        for x,y in items.iteritems()]
    return " ".join(s)

  def _printHtml(self,name,scrip,css,items):
    with open(name,'w') as htmlFile:
      htmlFile.write("""
<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8" >
    <title>cfg-browser</title>
    <script type="text/javascript" 
       src="http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js">
    </script>
    %(s)s
    %(css)s
  </head>
  <body>
    <div style="position: absolute; top: 0px; 
       right:0px; width: 100px; text-align:right;">
      <a href="javascript:void(0)" id="helpMouse"data-help="#help">help</a>
      <a href="mailto:susie.murphy@cern.ch?Subject=CfgBrowserhelp"
      target="_top">Contact</a>
    </div>
    <div id="help"> </div>
    <a id="hide" style="cursor:default; color:#CCCCCC;" 
      href="javascript:;" >Hide All</a>
    <br/><br/>
     <input type="checkbox" id="ShowFiles" value="File"/>Show module File.
  <br/>
   <form>
     <table border="1" cellpadding="3" cellspacing="1">Choose one:
       <tr>
         <td>%(items)s</td>
       </tr>
     </table>
   </form>
   <br/>
   <input type="submit" id="docSubmit" value="submit">
   <br/><br/>
   <div id="current"></div>
   <div id="attachParams"></div>
  </body>
</html>
      """%{"s":scrip,"css":css,"items":items})

  def _css(self, css, cssLocal):
    with open(css, 'w') as cssfile:
      cssfile.write("""
li {
  padding-left:0.8em;
}
ul {
  list-style-type:none;
  padding-left:0.1em;
}
.expand:before {
  content:'›';
  float: left;
  margin-right: 10px;
}
.expanded:before {
  content:'ˇ';
  float: left;
  margin-right: 10px;
}
.expand,.expanded {
  cursor:pointer;
}
/* colours of each. */
.Path,.EndPath{
  color:#192B33;
}
.Modules{ 
  color:#800080;
}
.SequenceTypes {
  color:#0191C8
}
.paramInner,.Types {
  color:#4B4B81;
}
.param {
  color: #9999CC;
  margin-left:0.8em;
  cursor:default; 
}
.value {
  color:#0000FF;
}
.type{
  color: #00CCFF;
}
/* Header for what's showing */
#current{
  color: #808080;
   font-size:14;
   text-decoration: underline; 
}
/* help settings */
#help {
  position: absolute;
  display: none;
  background: #ccc;
  border: 1px solid;
  width:250px;
}
h5 {
  font-style: normal;       
  text-decoration: underline; 
  margin:0px;
  font-size:10;
}
h6 {
  margin:0px;
}
#attachParams{
color:#192B33;
}
      """)
    return """<link href="%s" rel="stylesheet" \
           type="text/css"/>"""%(cssLocal)

  def _jqueryFile(self,name):
    with open(name, 'w')as jq:
      jq.write("""
$(document).ready(function(){ 
//Object used to get all details
var CURRENT_OBJ;
var showParams = false
var alreadyShowing = false
var topClass = "Top"
// ---
/*
 Functions used to abstract away name of function calls
  on the object (CURRENT_OBJ).
  (i.e. so function names can easily be changed.)
*/
function baseParams(inputs){
  return CURRENT_OBJ.getParameters(inputs);
}
function baseInnerParams(inputs, index){
  return CURRENT_OBJ.getInnerParams(inputs, index);
}
function baseType(inputs){
  return CURRENT_OBJ.getType(inputs);
}
function baseFile(inputs){
  return CURRENT_OBJ.getFile(inputs);
}
function baseTopFile(inputs){
  return CURRENT_OBJ.getTopFile(inputs);
}
// ---
/*
  Show something new!
*/

$(document).on('click', '#docSubmit', function(e){
  if(alreadyShowing){
    paramOptions(false)
    $(document.getElementsByTagName('body')).children('ul').remove();
  }
  var $elem = $("[name='docType']:checked")
  var docType =  $elem.attr("value");
  $("#current").html(docType)
  var gen = getGenericType(docType)
  var ty = docType
  
  if(gen != undefined){
    var gL =  gen.toLowerCase()
    if(gL=="modules"||gL=="types") var ty= gen;
  }
  var $LI = $(document.createElement('li')
   ).addClass("expand").addClass(ty).addClass(topClass);
  // create the object that we need
  //get the function we want and the lists
  var thefunction = $elem.attr("data-base");
  var list = $elem.attr("data-files").split(" ");
  var first = list[0]
  if(list.length >1){
    CURRENT_OBJ = window[thefunction](list[1], first)
   } 
  else{
    CURRENT_OBJ = window[thefunction](first)
  }
  docType= docType.toLowerCase()
  var showTypes = false
  showParams = true
  switch(docType){
    case "producer":
    case "consumer":
      showParams = false
      paramOptions(true)
      $LI.addClass("Modules")
      showTypes = true
      break;
    case "modules":
      $LI.addClass("Modules")
      showTypes = true
      //showParams = true
      break;
  }
  var $UL = addTopData(CURRENT_OBJ.getKeys(),$LI,showTypes)
  alreadyShowing = true;
  $UL.appendTo('#attachParams');
});
/*
 Used to add the top level data to html.
*/
function addTopData(data,$LI,types){
  var $UL = $(document.createElement('ul'));
  var doNormalFile = false;
  var files = document.getElementById("ShowFiles").checked
  if(files){
    try{
      baseTopFile(data[0])
    }
    catch(e){
      doNormalFile = true;
    }
  }
  for(var i=0; i < data.length;i++){
    var n = data[i]
    var t = n;
    if(types)t += " ("+baseType(n)+")"
    if(files){ 
      if(doNormalFile)var file = baseFile(n)
      else var file = baseTopFile(n)
      t += " ("+file+")"}
    $UL.append($LI.clone().attr("data-name",n).text(t));
  }
  return $UL;
}
/*
  Add option in html to show/hide parameters.
*/
function paramOptions(bool){
  if(!bool){
   $("#attachParams").empty()
   return
  }
  var lb= jQuery('<label/>', {
      for:"ShowParams"
  })
  jQuery('<input/>', {
      type:"checkbox",
      id:"ShowParams",
      name:"ShowParams",
      value:"ShowParams",
      autocomplete:"off"
    }).appendTo(lb)
  lb.append("Show Parameters")
lb.appendTo("#attachParams")
}
/*
  Retrieves and expands elements whose objects have multiple lists
  (i.e. onese who's data-base !=simpleData)
*/
$(document).on('click', '.Consumer.expand,.Producer.expand,'
+'.Path.expand,.EndPath.expand ',function(event){
  var LI = $(document.createElement('li')).addClass("expand");
  var allModules = CURRENT_OBJ.getModules($(this).attr('data-name'));
  var UL = addKids($(document.createElement("ul")), LI, allModules);
  $(this).append(UL); 
  $('#hide').css('color','');
  $('#hide').css('cursor','');
  event.stopPropagation();
});

/*
  Adds parameters onto objects.
*/
$(document).on('click','.Modules.expand,.SequenceTypes,.Types.expand'
, function(event){
  if(showParams){
    addParams(this);
  }
  event.stopPropagation();
});
/*
  Hides/Shows children from class param.
*/  
$(document).on('click', '.paramInner.expand, .paramInner.expanded',
   function(event){
   if($(this).children('ul').length ==0){
     // find parents
     var parents = findParents(this)
     var result = baseInnerParams(parents,parseInt(
                   $(this).attr("data-index")) )[1]
     addParams(this, result);
    }
    else{
      //children already added, so just hide/show.
      $(this).children('ul').toggle();
    }
    event.stopPropagation();
  });
/*
  Find the parents of a child.
*/
function findParents(child){
  var parents =[$(child).attr("data-name")]
  var theParent = $(child).attr("data-parent")
  while(theParent !=undefined){  
    var child = $(child).parent();
    if(child.prop("tagName")=="UL") continue;
    parents.unshift(child.attr("data-name"));
    theParent = child.attr("data-parent");
  }
  return parents
}
/*
  Helper function: returns filename appended onto val.
*/
function getFile(theName, val){
  var f = baseFile(theName)
  if(f)return val+=" ("+f+")"
  return val
}
/*
  Adds the provided kids to the list.
  Dont need to worry about adding parameter details.
  (it wont have any.)
*/
function addKids($UL, $LI, kids, classType){
  var fileChecked = document.getElementById("ShowFiles").checked
  for(var i=0; i < kids.length; i++){
    var tname = kids[i];
    var spec = baseType(tname)
    var gen = getGenericType(spec)
    var val = tname
    val += "("+spec+")"
    if(fileChecked)val = getFile(tname, val)
    $UL.append($LI.clone().attr("data-name", tname).addClass(gen).text(val));
  }
  return $UL;
}
/*
  Add params to the object. Object can be of any type 
  (normally module or param).
  Will be used by modules adding parameters, psets adding parameters.
*/
function addParams(obj, results){
  var fileChecked = document.getElementById("ShowFiles").checked
  var $LIBasic = $(document.createElement('li')).attr("class","param");
  var $LIExpand = $LIBasic.clone().attr("class","expand");
  var $span = $(document.createElement("span"));
  var $UL = $(document.createElement("ul"));
  var $objName = $(obj).attr('data-name');
  if(!results)
    var results = baseParams($objName)
  var parameters = results
  for(var i =0; i < parameters.length; i++){
    var all = parameters[i].slice();
    var theName = all.shift(); 
    if(all.length ==0)
     var typ = baseType(theName)
    else
      var typ = all.pop()
    var text = theName+" ("+typ+")"
    if(fileChecked) text = getFile(theName, text)
    if(typeof(all[0]) == "object"){
      // PSets
      var cloLI = doLI($LIExpand.clone(),theName,i,"paramInner",text,"")
      cloLI.attr("data-parent", $objName)
    }
    else if(baseParams(theName)){
      // Modules or sequences
      var cloLI = doLI($LIExpand.clone(),theName,i,getGenericType(typ),
                       text,"")
    }
    else{
      // Basic type, has no children
      var cloLI = $LIBasic.clone().attr("data-name", theName).attr(
      "data-index", i).text(theName);
      var cloLI= doLI($LIBasic.clone(),theName,i,"",theName,"")
      var value =""
      if(all.length)
        var value = all.shift()
      // formating so lots of strings look nicer
      if(value.indexOf(",")>-1){
         value = "<ul><li>"+value.replace(/,/g, "</li><li>")+"</li></ul>"
      }
      var add = " ("+typ+")"
      cloLI.append($span.clone().addClass("value").html(": "+value))  
      cloLI.append($span.clone().addClass("type").text(add))
      for(var p=0; p < all.length; p++){       
        cloLI.append($span.clone().addClass("type").text(" ("+all[p]+")"))
      } 
    } 
    $UL.append(cloLI);
  }
  $(obj).append($UL);
  $('#hide').css('color','')
  $('#hide').css('cursor','')
}
/*
  Helper function: Adds data to a LI.
*/
function doLI(LI,dataname,dataindex,classes,text,html){
   LI.attr("data-name", dataname).attr("data-index", dataindex).text(text);
   if(classes)LI.addClass(classes)
   if(html)LI.html(html)
   return LI
}
/*
  Box to show params has been clicked.
*/
$(document).on('click', '#ShowParams', function(e){
  if($(this).is (':checked')){
    showParams = true
  }
  else{
    $(this).next().hide()
    showParams = false
  }
});
/*
  Removes children from top level list elements.
*/
$(document).on('click', '#hide', function(e){
  //make sure not called when not needed.
  if($(this).css('cursor')!='default'){  
    var selec = $(".expanded."+topClass).children().hide()
    toggleExpand($(".expanded."+topClass ),e)
    $(this).css('color','#CCCCCC');
    $(this).css('cursor','default');
  }
});
/*
 Small info about each option.
*/
$('span[name="Info"]').mouseover(function(){
  var docType = $(this).attr("value").toLowerCase();
  var info;
  switch(docType){
    case "producer":
      info="What's produced by each module."
      break;
    case "consumer":
      info="What's consumed by each module."
      break;
    default:
      info ="List of "+ docType+"s."
  }
  $(this).attr("title", info);
});
/*
  More info about what's shown.
*/
$("#helpMouse").hover(function(e) {
    $($(this).data("help")).css({
        right: 250,
        top: e.pageY + 10
    }).stop().show(100);
    var title = "<h4>Info:</h4>"
     var expl = "<h5>Colour codes:</h5> <h6><ul><li class='Path'>pathName \
</li></ul><ul><li class='Modules'>Modules (e.g. EDProducer,\
 EDFilter etc)</li></ul>\
<ul><li class='Types'>Types (e.g. PSet)</li></ul>\
<ul><li class='param'>ParameterName:<span class='value'> value</span>\
<span class='type'>(type)</span></li></ul></h6>"
   var info ="<h5>The data</h5><h6>The headings you can choose \
from are what was collected from the config file.<br/><br/> \
Any change to the config file means having to run the script again \
and then refresh this page (if same output file was used).</h6>"
   $($(this).data("help")).html(title+expl+info);
}, function() {
    $($(this).data("help")).hide();
});
// Turn off any action for clicking help.
$('a#help').bind('click', function() {
   return false;
});
/*
  If parameter value is a list, hide/show the list on click.
*/
$(document).on('click', '.param',function(event){
  if($(this).find('ul').length >0){
    $(this).find('ul').toggle();
  }
  event.stopPropagation();
});
/*
  Removes children from expanded paths or modules.
*/
$(document).on('click', '.expanded',function(event){
  var c = $(this).children('ul');
  if(c.length >0){
    $(c).remove();  
  }
  event.stopPropagation();
});
// Toggles class names.
$(document).on('click','.expand, .expanded', function(event){
  toggleExpand(this, event);
});
/*
  Helper function toggles class type.
*/
function toggleExpand(me,event){
  $(me).toggleClass("expanded expand");
  event.stopPropagation();
}
});
// end of jquery
/*
Function to load the JSON files.
*/
function loadJSON(theName){
 return $.ajax({
    type: "GET",
    url: theName,
     beforeSend: function(xhr){
    if (xhr.overrideMimeType)
    {
      xhr.overrideMimeType("application/json");
    }
  },
    contentType: "application/json",
    async: false,
    dataType: "json"
  });
}
      """)
genericTypes={}

def doTypes(spec, generic):
  genericTypes[spec] = generic
def JSONFormat(d):
  import json
  return json.dumps(d)

if __name__ == "__main__":
  import sys, os, imp
  from optparse import OptionParser
  parser = OptionParser(usage="%prog <cfg-file> ")
  parser.add_option("-q", "--quiet",
                  action="store_true", dest="_quiet", default=False,
                  help="print minimal messages to stdout")
  parser.add_option("-o", "--html_file", dest="_htmlfile",
                    help="The output html file.", default="cfg-viewer.html")
  parser.add_option("-d", "--store_helper", dest="_helper_dir",default="cfgViewerJS",
                    help="Name of folder to store js,json and css files.")
  opts, args = parser.parse_args()
  if len(args)!=1:
    parser.error("Please provide one configuration file.")
  cfg = args[0]
  if(not os.path.isfile(cfg)):
    parser.error("File %s does not exist." % cfg)
  try:
    distBaseDirectory=os.path.abspath(
                      os.path.join(os.path.dirname(__file__),".."))
    if (not os.path.exists(distBaseDirectory) or 
       not "Vispa" in os.listdir(distBaseDirectory)):
      distBaseDirectory=os.path.abspath(
                        os.path.join(os.path.dirname(__file__),"../python"))
    if (not os.path.exists(distBaseDirectory) or 
       not "Vispa" in os.listdir(distBaseDirectory)):
      distBaseDirectory=os.path.abspath(os.path.expandvars(
                           "$CMSSW_BASE/python/FWCore/GuiBrowsers"))
    if (not os.path.exists(distBaseDirectory) or 
       not "Vispa" in os.listdir(distBaseDirectory)):
      distBaseDirectory=os.path.abspath(os.path.expandvars(
                          "$CMSSW_RELEASE_BASE/python/FWCore/GuiBrowsers"))
  except Exception:
    distBaseDirectory=os.path.abspath(
                       os.path.join(os.path.dirname(sys.argv[0]),".."))
  sys.path.insert(0,distBaseDirectory)
  from Vispa.Main.Directories import *
  distBinaryBaseDirectory=os.path.join(baseDirectory,"dist")
  sys.path.append(distBinaryBaseDirectory)
  from FWCore.GuiBrowsers.Vispa.Plugins.ConfigEditor import ConfigDataAccessor
  print "starting.."
  htmlF = opts._htmlfile
  htmldir= os.path.split(htmlF)[0]
  helper = opts._helper_dir
  if(htmldir):
    helperdir = "%s/%s"%(htmldir, helper)
  else:
    helperdir = helper
  if not os.path.exists(helperdir):os.makedirs(helperdir)
  unscheduled(cfg, htmlF, opts._quiet, helper+"/",helperdir+"/")
  #end = time.clock()
  #print end - start
