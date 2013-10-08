#!/usr/bin/env python
# -*- coding: latin-1 -*-
import re
import collections
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.SequenceTypes as seq

class unscheduled:
  def __init__(self,cfgFile,html,quiet,helperDir,fullDir):
    self._html = html
    self._serverName = os.path.join(os.path.split(html)[0],"EditingServer.py")
    self._quiet = quiet
    self._theDir= fullDir
    self._helperDir = helperDir
    self._mother,self._daughter ={},{}
    self._reg = re.compile("['>]")
    self._data,self._types,self._genericTypes={},{},{}
    self._dictP ="DictParent"
    self._modSeqP = "ModuleSeqParent"
    self._prodConsP=  "ProdConsumParent"
    self._parents = {
    "DictParent":{"creator":"dictCreate","simple": True},
    "ModuleSeqParent":{"creator":"modSeqCreate","simple": False},
    "ProdConsumParent":{"creator":"prodConCreate","simple": False}
    }
    for name,x in self._parents.iteritems():
      x["pfile"] = self._filenames(name)
      x["cfile"] = self._filenames(x["creator"])
    self._type = "%stypes.js"%(fullDir)
    self._allJSFiles =["types.js"]
    self.debug = False
    self._config= ConfigDataAccessor.ConfigDataAccessor()
    self._config.open(cfgFile)
    self._proceed()

  def _proceed(self):
    #self._filename= ""
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
    server(self._serverName)

  def _getData(self,objs):
    # i will loop around objs and keep adding things which are configFodlers
    calc =[]
    for each in objs:
      name = self._config.label(each)
      if(name =="esprefers"): self.debug = False
      else:
        self.debug = False
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

  # Get the data for items which are SequenceTypes
  def _doSequenceTypes(self,paths,namep):
    theDataFile,fullDataFile = self._calcFilenames(namep)
    topLevel,fullTopLevel = self._calcFilenames("top-"+namep)
    json = [topLevel,theDataFile]
    cap = namep.capitalize()
    bl={}
    types = False
    with open(fullDataFile,'w') as data:
      data.write("{")
      v = visitor(data)  
      for item in paths:
        if(not types):
          spec = self._checkType(item)
          self._saveData(spec,self._parents[self._modSeqP]["creator"],json) 
          types = True
        name = self._config.label(item)
        # Dont think we need to check for this here. 
        self._mothersDaughters(name,item)
        key = self._config.label(item)
        item.visit(v)
        bl[key]= getParamSeqDict(v._finalExit(),
                    self._config.fullFilename(item), "")
      data.write("}")
      with open(fullTopLevel, 'w') as other:
        other.write(JSONFormat(bl))

  # Check type of item and return the specofic type
  def _checkType(self,item):
    gen, spec = re.sub(self._reg, "", 
                             str(item.__class__)).split(".")[-2:]
    doTypes(spec,gen)
    return spec

  # find the mothers and daughters, storing them
  def _mothersDaughters(self,name, item):
    mo =self._config.motherRelations(item)
    dau = self._config.daughterRelations(item)
    if(mo):
      self._mother[name] = [self._config.label(i) for i in mo]
    if(dau):
      self._daughter[name] = [self._config.label(i) for i in dau]

  # Find data for objs which are not SequenceTypes
  def _doNonSequenceType(self,objs, globalType):
    everything={}
    always= types = False
    theDataFile, fullDataFile =self._calcFilenames(globalType)
    # For modules types can be diff 
    # so we always want to call the doTypes method
    if(globalType =="modules"):
      self._saveData(globalType.capitalize(),
                     self._parents[self._dictP]["creator"],
                     [theDataFile]) 
      always = types = True
    for item in objs:
      if(always or not types):
        spec = self._checkType(item)
        if(not types):
          self._saveData(spec,self._parents[self._dictP]["creator"],
                         [theDataFile]) 
          types = True
      name = self._config.label(item)
      self._mothersDaughters(name, item)
      if(isinstance(item,cms._Parameterizable)):
        out = getParameters(item.parameters_())
      elif(isinstance(item,cms._ValidatingListBase)):
        out = listBase(item)
      if(self.debug):
        print item, "file", self._config.fullFilename(item)
        print self._config.type(item)
        print item._filename
      everything[name] = getParamSeqDict(out,
                           self._config.fullFilename(item),
                           self._config.type(item))
    with open(fullDataFile,'w') as dataFile:
      dataFile.write(JSONFormat(everything))

  # Return json data file names
  def _calcFilenames(self,name):
    return self._filenames(name,"data-%s.json")

  # Return filenames
  def _filenames(self,name,option=""):
    if(option): basicName = option%(name)
    else: basicName = "%s.js"%(name)
    return (basicName, "%s%s"%(self._theDir,basicName))

  # write out mothers and daughters (producers/consumers).
  def _producersConsumers(self):
    if(not self._mother and not self._daughter):
      return  
    for name,theDict in {"producer":self._mother, 
                         "consumer":self._daughter}.iteritems():
      thedataFile , fulldataFile = self._calcFilenames(name)
      self._saveData(name.capitalize(),
                      self._parents[self._prodConsP]["creator"],
                      [thedataFile,self._calcFilenames("modules")[0]]) 
      with open(fulldataFile,'w') as moth:
        moth.write(JSONFormat(theDict))
  def _saveData(self,name,base,jsonfiles):
    jsonfiles = " ".join(["%s%s"%(self._helperDir,x)for x in jsonfiles])
    temp={}
    temp["data-base"] = base
    temp["data-files"] = jsonfiles
    self._data[name] = temp

  # Create objs and print out .js files for
  # each type of items we have.
  def _createObjects(self):
    base = "obj= Object.create(new %s(%s));"
    format="""
    function %s(%s){
      var obj;
      %s
      return obj;
    }
    """
    name = "data"
    for pname,x in self._parents.iteritems():
      simple = base%(pname,name)
      filename,fullfilename= x["cfile"]
      self._allJSFiles.append(filename)
      if(x["simple"]):
        paramName="modules"
        with open(fullfilename, 'w') as setUp:
          setUp.write(format%(x["creator"],paramName,self._load(
                                             name,paramName,simple)))
        continue
      secName = "topL"
      paramName=["modules","topLevel"]
      complexOne = base%(pname,"%s,%s"%(name, secName))
      with open(fullfilename, 'w') as setUp:
        setUp.write(format%(x["creator"],", ".join(paramName),
                      self._load(name,paramName[0],
                      self._load(secName, paramName[1],complexOne))))

  # return the .js code for loading json files in a string
  def _load(self,name,param,inner):
    return"""
      loadJSON(%s).done(function(%s){\n%s\n});
    """%(param,name,inner)

  # The parent class for non SequenceTypes
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
    search = """%(name)s.prototype.search%(key)s = function(reg,term,replce){
  return this.generalSearch(reg,term,replce,this.%(key)sKey);
}  """
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
    name = self._dictP
    data = self._parents[name]
    fileName, fullfileName= data["pfile"]
    for feature in dictFeatures:
      d = {"key": feature, "name": name}
      variables +=exVar%d
      functs +=exFunc%d
      if(feature == "Parameters"):
        functs +=extra%d
      else:
        functs += search%d
    self._allJSFiles.append(fileName)
    with open(fullfileName, 'w') as parent:
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
DictParent.prototype.getData = function(){
  return [this.data];
}
%(getterFunctions)s

/**
 * Gives the keys from desired dictionary.
* @returns {Array} all keys from the dictionary.
*/
%(name)s.prototype.getKeys = function(){
  return Object.keys(this.data);
}
%(name)s.prototype.generalSearch = function(reg,term,repl,feature, d){
  d = d || this.data;
  var matches ={}
  for (var key in d){
    var x= d[key][feature]
    if(x.indexOf(term)>-1){
      matches[key] = x.replace(reg,repl)
    }
  }
  return matches;
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
        "gen": typeName, "name":name})

  # Write out the class for SequenceTypes
  def _writeModSeqParent(self):
    f = """
/**
 * Gives the direct children
 * @param {String} a path name
 * @returns {Array} list of names of the the children.
 */
%(name)s.prototype.getModules = function(name){ 
  return this.topLevelData[name][this.ParametersKey];
}
%(name)s.prototype.getTopFile = function(key){
  return this.topLevelData[key][this.FileKey];
}
%(name)s.prototype.getData = function(){
  return [this.topLevelData, this.data];
}
%(name)s.prototype.searchType = function(reg,term,replce){
  return this.generalSearch(reg,term,replce,this.TypeKey, this.topLevelData);
}
%(name)s.prototype.searchFile = function(reg,term,replce){
  return this.generalSearch(reg,term,replce,this.FileKey, this.topLevelData);
}
"""
    self._complexBase(self._modSeqP, f)

  # Write out the class for producers and consumers.
  def _writeProdConsum(self):
    f= """
/**
 * Gives the direct children
 * @param {String} a path name
 * @returns {Array} list of names of the the children.
 */
%(name)s.prototype.getModules = function(name){ 
  return this.topLevelData[name];
}
%(name)s.prototype.getTopFile = function(key){
  return this.getFile(key);
}
%(name)s.prototype.getData = function(){
  return [this.topLevelData, this.data];
}
// Producer and consumer have different structure to rest.
// Dont have file and type with them..
// to get file and type we need to take each name, 
//look up the moduledata and find matches.
%(name)s.prototype.generalSearch = function(reg,term,repl, feature){
  var matches ={}
  for (var key in this.topLevelData){
    var x = this.data[key][feature]
    if(x.indexOf(term)>-1){
      matches[key] = x.replace(reg,repl)
    }
  }
  return matches;
}
%(name)s.prototype.typeSearch = function(reg,term,replce){
  return this.generalSearch(reg,term,replce,this.TypeKey);
}
%(name)s.prototype.fileSearch = function(reg,term,replce){
  return this.generalSearch(reg,term,replce,this.FileKey);
}
  """
    self._complexBase(self._prodConsP, f)

  def _complexBase(self,parentName, extra):
    name = parentName
    data = self._parents[name]  
    fileName, fullfilename= data["pfile"]
    self._allJSFiles.append(fileName)
    all = """
/* 
 Base object for thing of the type: 
 It also inherits from DictParent.           
*/                                          
function %(name)s(data,topLevel, nameList,indexList){ 
  this.data = data; 
  this.topLevelData=topLevel;// e.g. pathNames to module names 
  this.fixedNameList = nameList; // e.g.names of paths 
}
%(name)s.prototype = new %(dict)s(this.data);

%(name)s.prototype.getKeys = function(){
  return Object.keys(this.topLevelData)
}
      """+ extra
    with open(fullfilename, 'w') as parent:
      parent.write(all%{"name": name, "dict": self._dictP})

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
                    self._items(items), self._searchitems(items))

  def _scripts(self, js):
    x = """<script type="text/javascript" src="%s"></script>"""
    return "\n".join([x%(i) for i in js])

  def _items(self, items):
    l= """<option value=%(n)s data-base="%(d)s" 
    data-files="%(f)s"> %(n)s</option>"""
    s= [l%({"n":x,"f":y["data-files"],"d":y["data-base"]})
        for x,y in items.iteritems()]
    return " ".join(s)

  def _searchitems(self,items):
    # readded in name, for chrome.
    b = """<option name ="searchTerm1" value="%(name)s">%(name)s</option> """
    options = " ".join([b%({"name": x})for x in items])
    return """<form name="searchInput" onsubmit="return false;">
      <input type="text" id="searchWord"/>
      <select id="searchType">
       %s
      </select> 
      <span id="addSearch"> </span>
      <br/><br/>
       <input type="submit" value="Search" id="search"/></form> 
       """ %(options)


  def _printHtml(self,name,scrip,css,items, search):
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
    <a id="topOfPage"></a>
    <input type="submit" value="Edit " id="editMode"/>
    
    <br/> <br/>
    <div style="height: 2px; background-color: #A9A9A9; text-align: center">
      <span id="line" style="
  background-color: white; position: relative; top: -0.5em; color:#A9A9A9;">
    Normal Mode
     </span>
    </div>
    <br/>
    <div style="width: 100%%; overflow: hidden;">
      <div style ="float: left;margin-left: 10px;color:#A9A9A9;"> Pick
        <div style="width:400px;height:65px;border:2px dashed #A9A9A9;
          padding:10px;">
          <form onsubmit="return false;">
            <select id="showType">
             %(items)s
            </select>
            <input type="checkbox" id="ShowFiles" value="File"/>
            <span style="color:black">Show file names</span>
            <br/><br/>
            <input type="submit" id="docSubmit" value="submit">
          </form>
        </div>
      </div>
      <div style ="float: left;margin-left: 10px; color:#A9A9A9;"> Search
        <div style="width:400px;height:65px;border:2px dashed #A9A9A9;
          padding:10px;">
        %(search)s
        </div>
      </div>
    </div> 
    <script>
      document.getElementById("searchType").selectedIndex = -1;
    </script>
    <div id="posty"> </div>
    <div style="position: absolute; top: 0px; right:0px; width: 100px;
      text-align:right;">
       <a href="javascript:void(0)" id="helpMouse" data-help="#help">help</a>
       <a href="mailto:susie.murphy@cern.ch?Subject=CfgBrowserhelp"
        target="_top">Contact</a>
    </div>
    <div id="help"> </div>
    <br/>
    <div id="current"></div><br/>
    <div id="searchCount"></div>
    <div id="attachParams"></div>
    <div style="position: fixed; bottom: 50%%; right:0px; 
      width: 100px; text-align:right; border:2px solid #A9A9A9;
      opacity:0.4;background-color: #F0FFF0;">
      <a href="#topOfPage">Back to Top</a>
      <p>
        <a id="hide" style="cursor:default; color:#CCCCCC;" 
        href="javascript:;" >Hide All</a>
      </p>
    </div>
  </body>
</html> """%{"s":scrip,"css":css,"items":items, "search":search})

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
   font-size:12pt;
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
  font-size:9pt;
}
h6 {
  margin:0px;
  color:#666666;
}
#attachParams{
  color:#192B33;
}
em{
  color:#FFA500;
}
.cellEdit{
  border:2px dotted #A9A9A9;
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
var searching = false
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
  Current search only searches for top level things.
*/
$(document).on('click','#search', function(e){
   searching = true
   var first = $('#searchType :selected')
   if(first.length ==0){ 
     window.alert("Please chose a search type."); 
     return;}
   var doc = first.text()
   var next = $('#searchType2 :selected').text()
   var searchTerm = $('#searchWord').val()
   var $elem = $('option[value="'+doc+'"]')
   setCURRENT_OBJ($elem)
   var reg = new RegExp(searchTerm, 'g')
   var fin = "<em>"+searchTerm+"</em>"
   switch(next){
   case "File":
     $("#ShowFiles").prop('checked', true);
     var items = CURRENT_OBJ.searchFile(reg,searchTerm,fin)
     var matchCount= fileTypeSearch(doc,items,true)
     break;
   case "Type":
     var items = CURRENT_OBJ.searchType(reg,searchTerm,fin)
     var matchCount =fileTypeSearch(doc,items,false)
     break
   case "Name":
     var matchCount = easySearch(searchTerm,reg, doc, fin)
   }
   $("#searchCount").html(matchCount+ " found.")
  searching = false
});
/*
  Used when searching top level elements.
  i.e. Module names, path names etc.
  We dont need to delve into any dictionaries, we just use the keys.
*/
function easySearch(term,reg, doc, fin){
  var keys = CURRENT_OBJ.getKeys()
  var matches = keys.filter(function(e, i, a){return (e.indexOf(term)>-1)})
  var highlights = matches.map(function(e,i,a){
                               return e.replace(reg, fin)})
  addData(doc, highlights,matches)
  return matches.length
}

/*
  When searching type or file names of top level elements.
*/
function fileTypeSearch(doc,items,file){
  var newFunct = function(name){return items[name]}
  if(file){
    var backup = baseFile
    var backup2 = baseTopFile
    baseFile = newFunct  
    baseTopFile = newFunct
  }
  else {
    var backup = baseType
    baseType = newFunct
  }
  var matches = Object.keys(items)
  addData(doc, matches)
  if(file){
    baseFile = backup
    baseTopFile = backup2
  }
  else baseType = backup
  return matches.length
}

/*
  Show something new!
*/
$(document).on('click', '#docSubmit', function(e){
  var $elem =  $('#showType :selected')
  var docType =  $elem.attr("value");
 //get the function we want and the lists
 setCURRENT_OBJ($elem)
 addData(docType, CURRENT_OBJ.getKeys());
});

$(document).on('click', "[name='searchTerm1']", function(e){
//$(document).on('click', "#searchType option", function(e){
  $("#addSearch").empty()
  var name= $(this).text().toLowerCase();
  var sel= jQuery('<select/>', {
      id:"searchType2"
  })
  if(name =="modules"|| name == "producer"|| name == "consumer"){
    var li =  ["Name", "Type", "File"]
  }
  else var li= ["Name", "File"]
  for(var i in li){
    var x = li[i]
  jQuery('<option/>', {
      value:x,
      text:x
    }).appendTo(sel)
   }
  sel.appendTo("#addSearch");
});
$(document).on('click', '#normalMode', function(e){
  //remove all cell edit Divs
  goNormal($(this));
});

function goNormal(it){
  expandDisable = false;
  $(".cellEdit").replaceWith(function() { return $(this).contents(); });
  $("#save").remove()
  $("#line")[0].textContent = "Normal Mode"
  it.attr("value", "Edit mode");
  it.attr("id", "editMode")
}
var expandDisable = false;

$(document).on('click', '#editMode', function(e){
  // this is editing mode, turn off expanison
  var dataType = $('#showType :selected').attr("data-base")
  /*if(dataType != "dictCreate"){
    alert(
      "Right now, paths,endpaths,consumers and producers are non-editable.")
    return
  }*/
  expandDisable = true;
  $("#line")[0].textContent = "Edit Mode"
  $(this).after(jQuery('<input>', {
      type:"submit",
      value:"Save",
      id:"save"
    }))
  $(this).attr("value", "Finished Editing");
  $(this).attr("id", "normalMode")
  findAndAdd()
});

function findAndAdd(){
  // add editable div to everything showing.
  var todo = jQuery.makeArray($(".Top"));
  while(todo.length){
    var e = $(todo.pop())
    // we have the item.. lets get children
    var kids = jQuery.makeArray(e.children());
    // add divs to all children.
    while(kids.length){
      var k = $(kids.pop())
      var tag = k.prop("tagName")
      if(tag == "SPAN"){
         k.html( '<div class="cellEdit" contenteditable>'+k.html()+'</div>')
         continue
      }
      if(tag =="LI"){
        // add the div but dont get children
        $(k.contents()[0]).wrap('<div class="cellEdit" contenteditable />');
      }
      // check if have any children
      var moreKids = jQuery.makeArray(k.children());
      if(moreKids.length){
         kids = kids.concat(moreKids)
      }
    }
    // now do current one
    $(e.contents()[0]).wrap('<div class="cellEdit" contenteditable />');
  }
}
var $potential=[]
var parChildNames={}
$(document).on('click', '.cellEdit', function(e){
  // we're in edit so things have a div parent over the text.
  var $p = $(this).parent()
  var kidName = $p.attr("data-name");
  // now go up until greatest parent
  var classes = $p.attr("class")
  var parent = $p
  while(classes.indexOf("Top")==-1){
    parent = parent.parent()
    classes = parent.attr("class") || ""
  }
  var parentName = parent.attr("data-name");
  // doesnt matter if parent and child are the same.
  var parents = Object.keys(parChildNames)
  if(parents.indexOf(parentName)>-1){
    var kids = parChildNames[parentName] 
    if(kids.indexOf(kidName)==-1)kids.push($p)
  }
  else{
    parChildNames[parentName]=[$p]
  }
  $potential.push(parent)
  
});
function save(data, restData){
  var allChanged={}
  for(var i in $potential){
    var parent = $potential[i];
    // we have the parent.
    var oldParentName = $(parent).attr("data-name");
    if(data== restData){
      var allOldData= data[oldParentName]["Parameters"]
    }
    else{
      // we have one data for the parents and one for rest
     var allKids = data[oldParentName]["Parameters"]
     var tem ={}
     var childChanged = parChildNames[oldParentName];
     var oldToNew = chil(childChanged,parent, false);
     for (var y in allKids){
       var n = allKids[y]
       var allOldData = restData[n]
       var newpar = blah(oldToNew, n, allOldData, parent, true)
       tem[newpar] = allOldData
      }
      //change this TODO
      allChanged[oldParentName]= tem
      continue
    }
     var childChanged = parChildNames[oldParentName]
    var oldToNew= chil(childChanged,parent, true)
    var newpar = blah(oldToNew, oldParentName, allOldData, parent, false)
    allChanged[newpar] = allOldData
    
  }
  return allChanged
}

function blah(oldToNew, oldParentName, allOldData, parent){
    // pretend childChanged are kidsObjects
    // now we have all the children that were changed.
    var newpar = oldParentName
    if(Object.keys(oldToNew).indexOf(oldParentName)>-1){
      // parent is in
      var tData = oldToNew[oldParentName];
      if(tData["i"]==0 && tData["p"].length ==0){
        var newpar = tData["new"];
        delete oldToNew[oldParentName];
      }
    }
    // okay so here we have everything for this parent.
    // now need to loop through the old data and changed for new.
    if(Object.keys(oldToNew).length >0){
      loopData(allOldData,oldToNew,[oldParentName], [0]);
    }
  return newpar

}
//need to remember that current format is [name, parameters,type,optionalType]
function loopData(thelist, items, parents, pIndex){
  for(var x in thelist){
    var y = thelist[x]
    if(y instanceof Array){
      // okay if array, we have full line
      if(typeof y[0] == "string" && y[1] instanceof Array){
        // this is what we want.
       var theName = y[0]
       var rest = y[1]
       var types = [y[2]]
       if(y.length == 4)
         types.push(y[3])
       // okay, name of this is theName == parent
       // the index of this is x
       parents.unshift(theName)  
       pIndex.unshift(x)
       loopData(rest, items, parents, pIndex);
       parents.shift()
       pIndex.shift()
       rightOne(thelist,x,theName,items,parents,pIndex)
       for(var index in types){
         var item = types[index]
         rightOne(thelist,x,item,items,parents,pIndex)
       }
       continue
      }
      if(y[0] instanceof Array && y[1] instanceof Array){
        // y1 is a parameter
        // y2 is a parameter
        for(var e in y){
          loopData(y[e],items,parents,pIndex);
        }
       continue;
      }
      // we are here,
      //so we have something that does not have parameters in a list
      var theName = y[0]
      for(var index in y){
        if(index >0){
          parents.unshift(theName)  
          pIndex.unshift(x)
        }
        var item = y[index]
        rightOne(thelist,x,item,items,parents,pIndex)
        if(index >0){
          parents.shift()
          pIndex.shift()
        }
      }
      continue;
    }
    else {
      rightOne(thelist,x,y,items,parents,pIndex)
    }
  }
}
function rightOne(thelist,x,y,items,parents,pIndex){
  if(Object.keys(items).indexOf(y)>-1){
    // okay we know that we have a match, but are indexes the same?
    // index will be in the same list so we can use x
    if(x == items[y]["i"]){
     if(arrayCompare(parents, pIndex,items[y]["p"],items[y]["pIndex"] )){
        //okay we can be sure we are changing the right thing.
        if(thelist[x] instanceof Array){
          for(var ind in thelist[x]){
             var z = thelist[x][ind]
             if(y == z){
               thelist[x][ind] = items[y]["new"]
               break;
             } 
          }
        }
        else{
          thelist[x] = items[y]["new"]
        }
        delete items[y]
      }
    }
  }
}

function arrayCompare(a1, a1Index, a2, a2Index) {
    if (a1.length != a2.length)
        return false;
    for (var i = 0; i < a1.length; i++) {
        if (a1[i] != a2[i] || a1Index[i] != a2Index[i]) {
            return false;
        }
    }
    return true;
}

function chil (childChanged, parent, addParent){
  var oldToNew={}
  for (var x in childChanged){
      var child = childChanged[x]
      // we need to check if the child has been changed.
      var oldval = child.attr("data-name");
      if(child.prop("tagName")=="SPAN"){
           var newval = child.contents().text();
      }
      else{
         var newval = child.contents()[0].nodeValue;
      }
      //remove any enclosing brackets
      newval = newval.trim().replace(/^\((.+)\)$/, "$1");
      oldval = oldval.trim().replace(/^\((.+)\)$/, "$1");
      if(oldval==newval)continue
      else{
        //child has been changed.
        // we want to keep three pieces of data.
        //1. old name, new name
        //2. data-index
        //3. list of direct parents.
        var inner ={}
        inner["new"] = newval;
        inner["i"] = child.attr("data-index") || 0
        var allP = child.parents()
        var pNames =[]
        var pIndex =[]
        for(var y=0; y < allP.length; y++){
          var p = $(allP[y])
          var tag = p.prop("tagName")
          if(p.attr("data-name") == parent.attr("data-name")){
            if(addParent){
              pNames.push(parent.attr("data-name"))
              pIndex.push(parent.attr("data-index") || 0);
            }
            break;
          }
          else if(tag == "LI"){
            pNames.push(p.attr("data-name"));
            pIndex.push(p.attr("data-index") || 0);
          }
        }
        inner["p"] = pNames
        inner["pIndex"] = pIndex
        oldToNew[oldval]= inner
      }
    } // end of childchanged loop
 return oldToNew
}


$(document).on('click', '#save', function(e){
   var dataType = $('#showType :selected').attr("data-base")
  var allData = CURRENT_OBJ.getData();
  var oldData = allData[0]
  var changed={}
  // to save, if not in normal mode, change to normal mode.
  goNormal($("#normalMode"));
  if(allData.length >1){
    var top = allData[0];
    var rest = allData[1];
    var changed = save(top,rest)
  }
  else{
   var changed = save(allData[0], allData[0]);
  }
  var writeThis = changed
  var g = JSON.stringify(writeThis)
  //$('#svg_export_form > input[name=svg]').val("jhgjhg");
  $("#posty").empty()
  var form = jQuery('<form/>', {
      id:"edited_write",
      method:"POST",
      enctype: 'text/plain',
      style:"display:none;visibility:hidden",
    })
   jQuery('<input/>', {
      type:"hidden",
      name:"changed",
      value:g,
    }).appendTo(form)
    form.appendTo("#posty")
$('#edited_write').submit();
$potential=[]
parChildNames={}
 
});

function setCURRENT_OBJ($element){
  var thefunction = $element.attr("data-base");
  var list = $element.attr("data-files").split(" ");
  var first = list[0]
  if(list.length >1){
    CURRENT_OBJ = window[thefunction](list[1], first)
   } 
  else{
    CURRENT_OBJ = window[thefunction](first)
  }
}
function addData(docType, data, dataNames){
  var dataNames = dataNames || data;
  if(alreadyShowing){
    if(!searching)$("#searchCount").html("")
    $('#hide').css('color','#CCCCCC');
    $('#hide').css('cursor','default');
    paramOptions(false)
    $(document.getElementsByTagName('body')).children('ul').remove();
  }
  $("#current").html(docType)
  var gen = getGenericType(docType)
  var ty = docType
  
  if(gen != undefined){
    var gL =  gen.toLowerCase()
    if(gL=="modules"||gL=="types") var ty= gen;
  }
  var $LI = $(document.createElement('li')
   ).addClass("expand").addClass(ty).addClass(topClass);
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
  var $UL = addTopData(data,$LI,showTypes,dataNames)
  alreadyShowing = true;
  $UL.appendTo('#attachParams');
}

/*
 Used to add the top level data to html.
*/
function addTopData(data,$LI,types,dataName){
  var dataName = dataName || data;
  var $UL = $(document.createElement('ul'));
  var doNormalFile = false;
  var files = document.getElementById("ShowFiles").checked
  if(files){
    try{
      baseTopFile(dataName[0])
    }
    catch(e){
      doNormalFile = true;
    }
  }
  for(var i=0; i < data.length;i++){
    var n = dataName[i]
    var t = data[i];
    if(types)t += " ("+baseType(n)+")"
    if(files){ 
      if(doNormalFile)var file = baseFile(n)
      else var file = baseTopFile(n)
      t += " ("+file+")"}
    $UL.append($LI.clone().attr("data-name",n).html(t));
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
  Used when in edit mode to stop any items being added to the webpage.
  More items added after edit mode- they wont be editable, and when you
  click to edit something it loads its children.
*/
$(document).on('click','.expand, .expanded', function(event){
    if(expandDisable){
     event.stopImmediatePropagation();
    }
});
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
    var typ= !all.length ? baseType(theName): all.pop() 
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
      var valDataName = value
      if(value.indexOf(",")>-1){
         value = "<ul><li>"+value.replace(/,/g, ",</li><li>")+"</li></ul>"
      }
      var add = " ("+typ+")"
      cloLI.append($span.clone().addClass("value").attr("data-name",
        valDataName).html(value))  
      cloLI.append($span.clone().addClass("type").attr("data-name",
        add).text(add))
      for(var p=0; p < all.length; p++){       
        cloLI.append($span.clone().addClass("type").attr("data-name",
          " ("+all[p]+")").text(" ("+all[p]+")"))
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
    var selec = $(".expanded."+topClass).children("ul").hide()
    toggleExpand($(".expanded."+topClass ),e)
    $(this).css('color','#CCCCCC');
    $(this).css('cursor','default');
  }
});
/*
 Small info about each option.
*/
$('#showType option').mouseover(function(){
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
        top: e.pageY + 20,
        width: 300
    }).stop().show(100);
    var title = "<h6>(Read the README file!)</h6><h4>Info:</h4> "
    var expl = "<h5>Colour codes:</h5> <h6><ul><li class='Path'>pathName \
    </li></ul><ul><li class='Modules'>Modules \
    (e.g. EDProducer, EDFilter etc)</li></ul><ul><li class='Types'>\
    Types (e.g. PSet)</li></ul><ul><li class='param'>ParameterName:\
    <span class='value'> value</span><span class='type'>(type)</span>\
    </li></ul></h6>"
   var info ="<h5>The data</h5>\
   <h6>The headings you can choose from are what was collected from\
   the config file.<br/><br/> Any change to the config file means having\
   to run the script again and then refresh this page (if same output file\
   was used).</h6><br/>"
   var tSearch="<h5>Search</h5><h6>Currently can only search by listing \
   what items you would like to be searched, and then what part of \
   each item.<br/><br/> I.e. search the producers for certain names.\
   </h6><br/>"
   var problems = "<h5>HTML/JSON/JS issues</h5><h6>If content isn't \
   loading,or json files cannot be loaded due to browser security issues,\
   try runing a local server. Go to the dir that the html file is stored \
   in and type: <br/><span class='Types'>'python -m SimpleHTTPServer'\
   </span></h6><br/>"
   var editing = "<h5>Editing</h5><h6>In order to use the edit mode, \
   you need to run the EditingServer.py file, this will be in the same \
   directory as this html file.Then go to <span class='Types'>\
   http://localhost:8000/cfg-viewer.html.</span></h6><br/>"
   $($(this).data("help")).html(title+expl+info+tSearch+ problems+editing);
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
    if(!expandDisable){
  if($(this).find('ul').length >0){
    $(this).find('ul').toggle();
    }}
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

class server:
  def __init__(self, name):
    self.printServer(name)

  def printServer(self, name):
    with open(name, 'w') as f:
      f.write("""
#!/usr/bin/env python
import SimpleHTTPServer
import SocketServer
import logging
import cgi
import ast
import shutil
import os

PORT = 8000

class ServerHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

  def do_POST(self):
    clen = self.headers.getheader('content-length')
    rec = self.rfile.read(int(clen))
    # find the name and take it off.
    rec = rec[rec.find("{"):]
    self.writeFile(self.conversion(rec))
    print "finished writing"
    
  def conversion(self, json):
    # Need to convert my json into config
    # first thing = cms.{TheType}(_type, parameters)
    changed = json
    return changed

  def writeFile(self,changed):
    with open("edited.json", 'w') as f:
      f.write(changed)
    with open("edited.json", 'rb') as f:
      self.send_response(200)
      self.send_header("Content-Type", 'application/octet-stream')
      self.send_header("Content-Disposition", 
                       'attachment; filename="edited.json"')
      fs = os.fstat(f.fileno())
      self.send_header("Content-Length", str(fs.st_size))
      self.end_headers()
      shutil.copyfileobj(f, self.wfile)

Handler = ServerHandler

httpd = SocketServer.TCPServer(("", PORT), Handler)

print "using port", PORT
httpd.serve_forever()
      """)

if __name__ == "__main__":
  import sys, os, imp
  from optparse import OptionParser
  parser = OptionParser(usage="%prog <cfg-file> ")
  parser.add_option("-q", "--quiet",
                  action="store_true", dest="_quiet", default=False,
                  help="print minimal messages to stdout")
  parser.add_option("-o", "--html_file", dest="_htmlfile",
                    help="The output html file.", default="cfg-viewer.html")
  _helper_dir = "cfgViewerJS"
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
  helper = os.path.join(_helper_dir, "")
  if(htmldir):
    helperdir = os.path.join(htmldir, helper, "")
  else:
    helperdir = helper
  if not os.path.exists(helperdir):os.makedirs(helperdir)
  unscheduled(cfg, htmlF, opts._quiet, helper,helperdir)
