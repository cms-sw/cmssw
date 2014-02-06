#!/usr/bin/env python
# -*- coding: latin-1 -*-
import re
import collections
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.SequenceTypes as seq

class unscheduled:
  def __init__(self,cfgFile,html,quiet,helperDir,fullDir):
    self._html = html
    #self._serverName = os.path.join(os.path.split(html)[0],"EditingServer.py")
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
    self._config = ConfigDataAccessor.ConfigDataAccessor()
    self._config.open(cfgFile)
    self._computed = self._proceed(cfgFile)

  def _proceed(self, fileName):
    #self._filename= ""
    topObjs = self._config.topLevelObjects()
    if(len(topObjs)):
      self._getData(topObjs)
      ty = "genericTypes"
      with open(self._type, 'w') as f:
        f.write("var %s=%s"%(ty,genericTypes))
      self._createObjects()
      self._writeDictParent(ty)
      self._writeModSeqParent()
      self._writeProdConsum()
      fN = fileName.split("./")[-1].split(".")[0]
      JS = ["%s%s"%(self._helperDir,x)for x in self._allJSFiles]
      html(self._html,JS,self._data, self._theDir, self._helperDir,
      self._config.process().name_(), fN)
      
      return True
    else:
      print "---Nothing found for file %s."\
           " No computation done.---"%(fileName)
      return False

  def _getData(self,objs):
    # i will loop around objs and keep adding things which are configFolders
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
      v = visitor(data, self._config)  
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
                    self._config.fullFilename(item), self._config.type(item), "")
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
    theDataFile, fullDataFile = self._calcFilenames(globalType)
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
      if(isinstance(item,cms._TypedParameterizable)):
        oType = item.type_()
      else:
        oType =""
      everything[name] = getParamSeqDict(out,
                           self._config.fullFilename(item),
                           self._config.type(item), oType)
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

  
dictFeatures=["Parameters", "Type", "File", "oType"]
# Used to enforce dictionary in datafiles.
def getParamSeqDict(params, fil, typ, oType):
  d={}
  d[dictFeatures[0]] = params
  d[dictFeatures[1]] = typ
  d[dictFeatures[2]] = fil
  d[dictFeatures[3]] = oType
  return d

class visitor:
  def __init__(self, df, cfg):
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
    self.config = cfg

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
    #name = modObj.label_()
    name = self.config.label(modObj)
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
                          filename, specific, modObj.type_())
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
        name = self.config.label(value)
        #name = value.label_()
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
            d = getParamSeqDict([], "", specific, "")
            self._df.write(',"%s":%s'%(name,JSONFormat(d))) 
            self._done.append(name) 

  def leave(self, value):
    if(isinstance(value,cms._Module)):
      return
    elif(isinstance(value,cms._Sequenceable)):
      # now need to determine difference between 
      #ones which have lists and ones which dont
      if(isinstance(value, cms._ModuleSequenceType)):
        #name = value.label()
        name = self.config.label(value)
        if(name in self._oldNames):self._oldNames.remove(name)
        if(self._currentName == name):
          if(self._oldNames):
            self._currentName = self._oldNames.pop(0)
          else:
            self._currentName=""
        if(name not in self._done):
          generic,specific = self._getType(value)
          d = getParamSeqDict(self._seqs.pop(name), "", specific, "")
          self._df.write(',"%s":%s'%(name,JSONFormat(d)))
          self._done.append(name)  
        self._seq -=1
        if(self._seq==0): self._innerSeq = False;

class html:
  def __init__(self,name,js,items,theDir, helperDir, pN,pFN):
    jqName = "%scfgJS.js"%(theDir)
    jqLocal = "%scfgJS.js"%(helperDir)
    css = "%sstyle.css"%(theDir)
    cssL = "%sstyle.css"%(helperDir)
    js.insert(0,jqLocal)
    self._jqueryFile(jqName, pN,pFN)
    self._printHtml(name,self._scripts(js),self._css(css, cssL),
                    self._items(items), self._searchitems(items))

  def _scripts(self, js):
    x = """
<script type="text/javascript" src="%s"></script>"""
    return "\n".join([x%(i) for i in js])

  def _items(self, items):
    l= """
<option value=%(n)s data-base="%(d)s" data-files="%(f)s"> %(n)s</option>"""
    s= [l%({"n":x,"f":y["data-files"],"d":y["data-base"]})
        for x,y in items.iteritems()]
    return " ".join(s)

  def _searchitems(self,items):
    b = """<option value="%(name)s" data-base="%(d)s">%(name)s</option> """
    return "\n ".join([b%({"name": x, "d":y["data-base"]})
           for x,y in items.iteritems()])

  def _printHtml(self,name,scrip,css,items, search):
    with open(name,'w') as htmlFile:
      htmlFile.write("""<!DOCTYPE html>\n<html>\n  <head>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8" >
    <title>cfg-browser</title>\n    <script type="text/javascript" 
       src="http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js">
    </script>\n    %(s)s\n    %(css)s\n  </head>\n  <body>
      <div class="topBox topSpec">
      <input type="submit" value="Edit" id="editMode" autocomplete="off"
        disabled/>\n      <div class="helpLinks">
        <a href="#" class="nowhere" id="helpMouse" data-help="#help">help
        </a></br/>
        <a href="mailto:susie.murphy@cern.ch?Subject=CfgBrowserhelp" 
         target="_top">Contact</a>\n      </div>\n      <div id="help"> </div>
      <span id="mode">Normal Mode</span>
      <div class="topBox hideTopBox">hide</div>\n    </div>
    <a id="topOfPage"></a>\n    <br/><br/><br/>\n    <div class="outerBox">
      <div class="boxTitle"> Pick\n        <div class="dottedBorder">
          <form onsubmit="return false;">\n            <select id="showType">
             %(items)s\n            </select>\n            <br/><br/>
            <input type="submit" id="docSubmit" value="submit">
          </form>\n        </div>\n      </div>
      <div class="boxTitle"> Pick\n        <div class="dottedBorder">
          <form onsubmit="return false;">
            <input type="text" id="searchWord"/>
            <select id="searchType">\n              %(search)s
           </select> \n            <span id="addSearch"> </span>
            <br/><br/>
            <input type="submit" value="Search" id="search"/>
            <input type="checkbox" id="rmHilight" value="File"/>
            <span>Don't show highlights</span>\n          </form> 
        </div>\n      </div>\n    </div> \n    <br/>
    <input type="checkbox" id="ShowFiles" value="File"/>
    <span>Show file names</span>\n    <br/><br/>\n    <script>
      document.getElementById("searchType").selectedIndex = -1;\n    </script>
    <div id="posty"> </div>\n    <br/>\n    <div id="current"></div>
    <br/>\n    <div id="searchCount"></div>\n    <div id="attachParams"></div>
    <div class="rightScroll">\n      <a href="#topOfPage">Back to Top</a>
      <p>\n        <a id="hide" href="javascript:;" >Hide All</a>\n      </p>
    </div>\n  </body>\n</html>\n"""%{
      "s":scrip,"css":css,"items":items, "search":search})

  def _css(self, css, cssLocal):
    with open(css, 'w') as cssfile:
      cssfile.write("""
*{\n   font-family:"Lucida Grande", Tahoma, Arial, Verdana, sans-serif;\n}
.topBox{\n  opacity:0.9;\n  background-color:#dfdfdf;\n}\n.topSpec{
    position:fixed;\n  top:0;\n  left:0;\n  width:100%;\n  height:2.5em;
  border-radius:.9em;\n}\n.hideTopBox, .showTopBox{\n  position:absolute;
  top:2.5em;\n  right:2%;\n  width:auto;\n  height:auto;\n  padding: 0.2em;
  font-weight:bold;\n  cursor:pointer;\n  border-radius:.3em;\n}\n.negative{
    position:absolute;\n    right:0;\n    text-decoration:none;
    font-weight:bold;\n    color:#565656;\n}\n#mode{\n  position:absolute;
  right:50%; \n  color:#808080;\n  font-size:12pt;\n}\n#editMode,#normalMode{
  position:absolute;\n  left:.3em;\n  top:.5em;\n}\n#save{
  position:absolute;\n  top:.5em;\n}\n.helpLinks{\n  position:absolute;
  right:1.3em;\n  width:6.2em;\n  text-align:right;\n}
/* divs for the Pick and Search boxes.*/\n.outerBox{\n  width:100%;
  overflow:hidden;\n}\n.boxTitle{\n  float:left;\n  margin-left:.6em;
  color:#A9A9A9;\n}\n.dottedBorder{\n  width:25em;\n  height:4em;
  border:.1em dashed #A9A9A9;\n  padding:.6em;\n}\n/* -- */
/* Right scroll box. */\n.rightScroll{\n  position:fixed;\n  bottom:50%;
  right:0; \n  width:6.2em;\n  text-align:right;\n  border:.1em solid #A9A9A9;
  opacity:0.4;\n  background-color:#F0FFF0;\n}\n#hide{\n  cursor:default;
  opacity:0.2;\n}\nli{\n  padding-left:.8em;\n}\nul{\n  list-style-type:none;
  padding-left:.1em;\n}\n/* Icons before list items. */\n.expand:before{
  content:'›';\n}\n.expanded:before{\n  content:'ˇ';\n}
.expand:before,.expanded:before{\n  float:left;\n  margin-right:.6em;\n}
.expand,.expanded{\n  cursor:pointer;\n}\n/* colours of each. */
.Path,.EndPath{\n  color:#192B33;\n}\n.Modules{ \n  color:#800080;\n}
.SequenceTypes{\n  color:#0191C8\n}\n.paramInner,.Types{\n  color:#4B4B81;\n}
.param{\n  color:#9999CC;\n  margin-left:.8em;\n  cursor:default; \n}\n.value{
  color:#0000FF;\n}\n.type{\n  color:#00CCFF;\n}
/* Header for what's showing */\n#current{\n  color:#808080;\n  font-size:12pt;
  text-decoration:underline; \n}\n/* help settings */\n#help{
  position:absolute;\n  display:none;\n  background:#ccc;\n  border:.1em solid;
  width:19em;\n  right:15em;\n  top:1.2em;\n}\nh5{\n  font-style:normal;
  text-decoration:underline; \n  margin:0;\n  font-size:9pt;\n}\nh6{
  margin:0;\n  color:#666666;\n}\n#attachParams{\n  color:#192B33;\n}\nem{
  color:#FFA500;\n}\n.cellEdit{\n  border:.1em dotted #A9A9A9;\n  cursor: text;
}\n""")
    return """<link href="%s" rel="stylesheet" \
           type="text/css"/>"""%(cssLocal)

  def _jqueryFile(self,name, pN,pFN):
    with open(name, 'w')as jq:
      jq.write("""
\n$(document).ready(function(){ \n//Object used to get all details
var processName="%s";\nvar processFileName ="%s";
var CURRENT_OBJ;\nvar searchShowing= false;\nvar showParams = false
var alreadyShowing = false\nvar topClass = "Top"\nvar searching = false
var expandDisable = false;\nvar hideVisible=false //nothing is expanded.
// ---\n/*\n Functions used to abstract away name of function calls
  on the object (CURRENT_OBJ).
  (i.e. so function names can easily be changed.)\n*/
function baseParams(inputs){\n  return CURRENT_OBJ.getParameters(inputs);\n}
function baseInnerParams(inputs, index){
  return CURRENT_OBJ.getInnerParams(inputs, index);\n}
function baseType(inputs){\n  return CURRENT_OBJ.getType(inputs);\n}
function baseFile(inputs){\n  return CURRENT_OBJ.getFile(inputs);\n}
function baseTopFile(inputs){\n  return CURRENT_OBJ.getTopFile(inputs);\n}
// --- Show some inital data operations ---\n/*\n  Show something new!\n*/
$(document).on('click', '#docSubmit', function(e){
  var $elem =  $('#showType :selected')
  //get the function we want and the lists\n  setCURRENT_OBJ($elem)\n     
  addData($elem.attr("value"), CURRENT_OBJ.getKeys());\n  searchShowing= false;
});\n/*\n  Add data to the html.\n*/
function addData(docType, data, dataNames){
  $("#editMode").removeAttr("disabled"); \n var dataNames = dataNames || data;
  if(alreadyShowing){\n    //need to click cancel if its active.
    goNormal($("#normalMode"));\n    if(!searching)$("#searchCount").html("")
    invisibleHide()\n    paramOptions(false)
    $(document.getElementsByTagName('body')).children('ul').remove();\n  }
  $("#current").html(docType)\n  var gen = getGenericType(docType)
  var ty = docType\n  if(gen != undefined){\n    var gL =  gen.toLowerCase()
    if(gL=="modules"||gL=="types") var ty= gen;\n  }
  var $LI = $(document.createElement('li')
   ).addClass("expand").addClass(ty).addClass(topClass);
  docType= docType.toLowerCase()\n  var showTypes = false\n  showParams = true
  switch(docType){\n    case "producer":\n    case "consumer":
      showParams = false\n      paramOptions(true)
      $LI.addClass("Modules")\n      showTypes = true\n      break;
    case "modules":\n      $LI.addClass("Modules")\n      showTypes = true
      //showParams = true\n      break;\n  }
  var $UL = addTopData(data,$LI,showTypes,dataNames)\n  alreadyShowing = true;
  $UL.appendTo('#attachParams');\n}\n/*
 Used to add the top level data to html.\n*/
function addTopData(data,$LI,types,dataName){
  var dataName = dataName || data;
  var $UL = $(document.createElement('ul'));\n  var doNormalFile = false;
  var files = document.getElementById("ShowFiles").checked\n  if(files){
    try{\n      baseTopFile(dataName[0])\n    }\n    catch(e){
      doNormalFile = true;\n    }\n  }\n  for(var i=0; i < data.length;i++){
    var n = dataName[i]\n    var t = data[i];
    if(types)t += " ("+baseType(n)+")"\n    if(files){ 
      if(doNormalFile)var file = baseFile(n)
      else var file = baseTopFile(n)\n      t += " ("+file+")"}
    $UL.append($LI.clone().attr("data-name",n).html(t));\n  }\n return $UL;\n}
// --- end of inital showing data operations ---\n
// --- search operations ---\n
//$(document).on('click', "[name='searchTerm1']", function(e){
//$(document).on('click', "#searchType option", function(e){
$(document).on('click', '#searchType', function(e){\n  $("#addSearch").empty()
  var tname= $(this).text().toLowerCase();\n  var sel= jQuery('<select/>', {
      id:"searchType2"\n  })
  var li = (tname =="modules"|| tname == "producer"|| tname == "consumer")?
           ["Name", "Type", "File"]: ["Name", "File"]\n  for(var i in li){
    var x = li[i]\n  jQuery('<option/>',{\n      value:x,\n      text:x
    }).appendTo(sel)\n   }\n  sel.appendTo("#addSearch");\n});\n/*
  Current search only searches for top level things.\n*/
$(document).on('click','#search', function(e){\n   searching = true
   var first = $('#searchType :selected')\n   if(first.length ==0){ 
     window.alert("Please chose a search type."); \n     return;\n   }
   var doc = first.text()\n   var searchTerm = $('#searchWord').val()
   var $elem = $('option[value="'+doc+'"]')\n   setCURRENT_OBJ($elem)
   var reg = new RegExp(searchTerm, 'g')
   if($('#rmHilight').prop('checked'))\n     var fin = searchTerm\n   else
     var fin = "<em>"+searchTerm+"</em>"
   switch($('#searchType2 :selected').text()){\n   case "File":
     $("#ShowFiles").prop('checked', true);
     var items = CURRENT_OBJ.searchFile(reg,searchTerm,fin)
     var matchCount = fileTypeSearch(doc,items,true)\n     break;
   case "Type":\n     var items = CURRENT_OBJ.searchType(reg,searchTerm,fin)
     var matchCount = fileTypeSearch(doc,items,false)\n     break
   case "Name":\n     var matchCount = easySearch(searchTerm,reg, doc, fin)
   }\n   $("#searchCount").html(matchCount+ " found.")\n  searching = false;
  searchShowing= true;\n});\n/*\n  Used when searching top level elements.
  i.e. Module names, path names etc.
  We dont need to delve into any dictionaries, we just use the keys.\n*/
function easySearch(term,reg, doc, fin){\n  var keys = CURRENT_OBJ.getKeys()
  var matches = keys.filter(function(e, i, a){return (e.indexOf(term)>-1)})
  var highlights = matches.map(function(e,i,a){
                               return e.replace(reg, fin)})
  addData(doc, highlights,matches)\n  return matches.length\n}\n/*
  When searching type or file names of top level elements.\n*/
function fileTypeSearch(doc,items,file){
  var newFunct = function(name){return items[name]}\n  if(file){
    var backup = baseFile\n    var backup2 = baseTopFile
    baseFile = newFunct  \n    baseTopFile = newFunct\n  }\n  else {
    var backup = baseType\n    baseType = newFunct\n  }
  var matches = Object.keys(items)\n  addData(doc, matches)\n  if(file){
    baseFile = backup\n   baseTopFile = backup2\n  }\n  else baseType = backup
  return matches.length\n}\n// --- end of search operations ---\n
// --- edit operations ---\n// variables needed:
// will contain the highest parents, where something *might*
// be changed in its children.\nvar $potential=[]
// dict, keys are parentNames and values are lists of children changed
// under that parent\nvar parChildNames={}\n
$(document).on('click', '#editMode', function(e){\n  if(searchShowing)
    var dataType = $('#searchType :selected').attr("data-base")
  else var dataType = $('#showType :selected').attr("data-base")
  // Temp restriction of editing SequenceTypes e.g. Paths,EndPaths etc.
  if(dataType=="modSeqCreate"){
    window.alert("At the moment editing for SequenceTypes"+
                 " has been disabled.");\n    return;\n  }
  else if (dataType == "prodConCreate"){
    window.alert("Sorry, it is currently not possible to "+
                "edit producers and consumers.");\n    return;\n  }
  if(CURRENT_OBJ == undefined)return;
  // this is editing mode, turn off expansion\n  expandDisable = true;
  $("#mode")[0].textContent = "Edit Mode"\n  $(this).attr("value", "Cancel");
    var l = ($(this).width()/ parseFloat($("body").css("font-size"))+2.5)+"em"
    $(this).after(jQuery('<input>', {\n      type:"submit",
      value:"Save",\n      id:"save"\n    }).css({\n       left:l\n}));
  $(this).attr("id", "normalMode")\n  makeEditable(searchShowing);
  searchShowing = false;\n});\n/*
  Adds in div elements to make cells editable.\n*/
function makeEditable(rmSearch){\n  // add editable div to everything showing.
  var todo = jQuery.makeArray($(".Top"));\n  while(todo.length){
    var e = $(todo.pop())\n    if(rmSearch)
     e.html(e.html().replace(/(<em>|<\/em>)/g, ""))
    // we have the item.. lets get children
    var kids = jQuery.makeArray(e.children());
    // add divs to all children.\n    while(kids.length){
      var k = $(kids.pop())\n      var tag = k.prop("tagName")\n
      if(tag == "SPAN"){\n        if(k.prop("class")!="type")
          k.html('<div class="cellEdit" contenteditable>'+k.html()+'</div>')
        continue\n      }\n      if(tag =="LI"){
        // add the div but dont get children
        $(k.contents()[0]).wrap('<div class="cellEdit" contenteditable />');
      }\n      // check if have any children
      var moreKids = jQuery.makeArray(k.children());
      if(moreKids.length){\n        kids = kids.concat(moreKids)\n      }
    }\n    // now do current one
    $(e.contents()[0]).wrap('<div class="cellEdit" contenteditable />');\n  }
}\n/*\n  A cell which is editable has been clicked.\n*/
$(document).on('click', '.cellEdit', function(e){
  // we're in edit so things have a div parent over the text.
  // so get parent to get the thing we want.
  var $itemChanged = $(this).parent()
  // now go up until we find greatest parent(item with class==topClass)
  var classes = $itemChanged.attr("class")\n  var $parent = $itemChanged 
  while(classes.indexOf(topClass)==-1){\n    $parent = $parent.parent();
    classes = $parent.attr("class") || "";\n  }
  var parentName = $parent.attr("data-name");
  // doesnt matter if parent and child are the same.
  var parents = Object.keys(parChildNames)
  if(parents.indexOf(parentName)>-1){\n    // already been edited.
    var kids = parChildNames[parentName] 
    if(kids.indexOf($itemChanged.attr("data-name"))==-1){
      kids.push($itemChanged)\n    }\n  }\n  else{
    parChildNames[parentName]=[$itemChanged]\n  }\n  $potential.push($parent)
});\n//--- end of edit operations ---\n// --- start of save operations ---\n/*
 Save any changes.\n*/\n$(document).on('click', '#save', function(e){
  var allData = CURRENT_OBJ.getData();\n  var oldData = allData[0]
  goNormal($("#normalMode"));\n  if(allData.length >1){
    var top = allData[0];\n    var rest = allData[1];
    var changed = save(deepCopy(top),deepCopy(rest))\n  }\n  else{
    var changed = save(deepCopy(allData[0]));\n  }
  if(Object.keys(changed).length ==0){
    window.alert("Nothing was changed.");\n    $potential=[]
    parChildNames={}\n    return \n  }
  //$('#svg_export_form > input[name=svg]').val("jhgjhg");
  changed["processName"]= processName
  changed["processFileName"]= processFileName\n  $("#posty").empty()
  var form = jQuery('<form/>', {\n    id:"edited_write",\n    method:"POST",
    async: "false",\n    enctype: "multipart/form-data",
    style:"display:none;visibility:hidden",\n  })\n  jQuery('<input/>', {
    type:"hidden",\n    name:"changed",\n    value:JSON.stringify(changed),
  }).appendTo(form)\n  form.appendTo("#posty")\n  $('#edited_write').submit();
  $potential=[]\n  parChildNames={}\n});\n/*
  Go to normal viewing mode. Discard any changes.\n*/
$(document).on('click', '#normalMode', function(e){
  //remove all cell edit Divs\n  goNormal($(this));\n  //$potential =[]
  //parChildNames={}\n});\n/*\n Couldn't find built-in deep clone or copy.\n*/
function deepCopy(obj){\n  if(obj instanceof Array){\n    var copy=[]
    obj.forEach(function(x,i,a){copy[i]= deepCopy(x)})\n    return copy\n  }
  else if(obj instanceof Object){\n    var copy ={}\n    for (key in obj){
      copy[key] = deepCopy(obj[key])\n    }\n    return copy\n  }\n return obj
}\n\n/*\n Returns a dict of everything that has changed.\n*/
function save(data, restData){\n  var allChanged={}\n  var done=[]
  for(var i in $potential){\n    var parent = $potential[i];
    // we have the parent.
    var oldParentName = $(parent).attr("data-name");
    //bit iffy, can have same parent name multiple times? todo
    if(done.indexOf(oldParentName)>-1) continue\n    done.push(oldParentName)
    if(typeof restData =='undefined'){
      var allDict = deepCopy(data[oldParentName])
      var allOldData = allDict["Parameters"]
      var childChanged = parChildNames[oldParentName]
      var oldToNew= chil(childChanged,parent, true)
      if(oldToNew.length ==0) continue
      var newpar = blah(oldToNew, oldParentName, allOldData, parent, false)
      tempDict = allDict\n      tempDict["Parameters"] =  allOldData
      allChanged[newpar] = tempDict\n      if(oldToNew.length>0){
        window.alert("Something went wrong, I did not find all changes.");
      }\n    }\n    else{
      // we have one data for the parents and one for rest
      var allDict = deepCopy(data[oldParentName])
      var allKids = allDict["Parameters"]\n      var tem ={}
      var childChanged = parChildNames[oldParentName];
      var oldToNew = chil(childChanged,parent, false);
      //if(Object.keys(oldToNew).length ==0) continue
      if(oldToNew.length ==0) continue\n      for (var y in allKids){
        var n = allKids[y]\n        var allOld2 = deepCopy(restData[n])
        var allOldData = allOld2["Parameters"]
        // blah chanhes allOldData
        var newpar = blah(oldToNew, n, allOldData, parent, true)
        allOld2["Parameters"]= allOldData\n        tem[newpar] = allOld2
      }\n      var newParentName = $(parent).contents()[0].nodeValue
      //allChanged[newParentName]= tem\n      tempDict = allDict
      tempDict["Parameters"] =  tem\n     allChanged[newParentName] = tempDict
      continue\n    }\n  }\n  return allChanged\n}\n/*
 These three functions are used for getting data from
 a structure such as [{},{},{}]\n*/\nfunction getValue(list, key){\n var re=[]
  for(x in list){\n    di = list[x]\n    k = Object.keys(di)[0]
    if(k==key)re.push(di[k])\n  }\n  return re\n}\nfunction getKeys(list){
  var keys = []\n  for (x in list)\n    keys.push(Object.keys(list[x])[0])
  return keys\n}\nfunction deleteValue(list, k,d){\n  var inde =-1
  var found = false\n  for(var x in list){\n    var di = list[x]
    var key = Object.keys(di)[0]\n    if(key==k && di[key]==d){
      found = true\n      inde = x\n      break\n    }\n  }
  if(found)list.splice(inde,1)\n}\n/*
  allOldData will be changed with the new data.\n*/
function blah(oldToNew, oldParentName, allOldData, parent){
    var newpar = oldParentName\n    var keys = getKeys(oldToNew)
    if(keys.indexOf(oldParentName)>-1){\n      // parent is in
      var tDatas = getValue(oldToNew,oldParentName);
      for (var x in tDatas){\n        var tData = tDatas[x]
        if(tData["i"]==0 && tData["p"].length ==0){
          var newpar = tData["new"];\n          // this isnt finding tData
          deleteValue(oldToNew, oldParentName,tData);\n      }\n     }\n    }
    // okay so here we have everything for this parent.
    // now need to loop through the old data and changed for new.
    if(oldToNew.length >0){
      loopData(allOldData,oldToNew,[oldParentName], [0]);\n    }
  return newpar\n}\n//need to remember that current format is 
//[name, parameters,type,optionalType]\n/*\n  Loop around the data, \n*/
function loopData(thelist, items, parents, pIndex){\n  for(var x in thelist){
    var y = thelist[x]\n    if(y instanceof Array){
      // okay if array, we have full line
      if(typeof y[0] == "string" && y[1] instanceof Array){
        // this is what we want.\n       var theName = y[0]
       var rest = y[1]\n       var types = [y[2]]\n       if(y.length == 4)
         types.push(y[3])\n       // okay, name of this is theName == parent
       // the index of this is x\n       parents.unshift(theName)  
       pIndex.unshift(x)\n       loopData(rest, items, parents, pIndex);
       parents.shift()\n       pIndex.shift()
       rightOne(thelist,x,theName,items,parents,pIndex)
       for(var index in types){\n         var item = types[index]
         rightOne(thelist,x,item,items,parents,pIndex)\n       }
       continue\n      }
      if(y[0] instanceof Array && y[1] instanceof Array){
        // y1 is a parameter\n        // y2 is a parameter
        for(var e in y){\n          loopData(y[e],items,parents,pIndex);
        }\n       continue;\n      }\n      // we are here,
      //so we have something that does not have parameters in a list
      var theName = y[0]\n      for(var index in y){\n        if(index >0){
          parents.unshift(theName)  \n          pIndex.unshift(x)\n        }
        var item = y[index]
        rightOne(thelist,x,item,items,parents,pIndex)\n        if(index >0){
          parents.shift()\n          pIndex.shift()\n        }\n      }
      continue;\n    }\n    else {
      rightOne(thelist,x,y,items,parents,pIndex)\n    }\n  }\n}\n/*
  Helper function. We have a match, now change old values for new values.\n*/
function rightOne(thelist,x,y,items,parents,pIndex){
  if(getKeys(items).indexOf(y)>-1){
    // we know that we have a match, but are indexes the same?
    // index will be in the same list so we can use x
    var dis = getValue(items, y)\n    for (inde in dis){
      var di= dis[inde]\n      if(x == di["i"]){
        if(arrayCompare(parents, pIndex,di["p"],di["pIndex"] )){
          //okay we can be sure we are changing the right thing.
          if(thelist[x] instanceof Array){
            for(var ind in thelist[x]){\n              var z = thelist[x][ind]
              if(y == z){\n                thelist[x][ind] = di["new"]
                break;\n              } \n            }\n          }
          else{\n            thelist[x] = di["new"]\n          }
          deleteValue(items,y,di)\n        }\n      }\n    }\n  }\n}\n/*
  Returns true if arrays a1 is identical to a2 and
  a1Index is identical to a2Index.\n*/
function arrayCompare(a1, a1Index, a2, a2Index) {
    if (a1.length != a2.length)return false;
    for (var i = 0; i < a1.length; i++) {
        if (a1[i] != a2[i] || a1Index[i] != a2Index[i]) {
            return false;\n        }\n    }\n    return true;\n}\n/*
  Take in the children that might have been changed.
  If something has been changed, adds to a dictionary
  dictionary format is  [data-name]={
  "new": newvalue, "i":data-index, "p":[parentNames],
  "pIndex": [indicies of parents]}
  TODO, need a new format, what if children have 
  the same data-name and both are changed?\n*/
function chil (childChanged, parent, addParent){\n  var oldToNew=[]
  for (var x in childChanged){\n      var child = childChanged[x]
      // we need to check if the child has been changed.
      var oldval = child.attr("data-name");
      var isSpan = child.prop("tagName") == "SPAN"\n     var newval = isSpan ?
       child.contents().text(): child.contents()[0].nodeValue;
      //remove any enclosing brackets
      newval = newval.trim().replace(/\(.*\)$/, "");
      oldval = oldval.trim().replace(/\(.*\)$/, "");
      if(oldval==newval)continue\n      else{
        //child has been changed.\n        //we want to keep 4 pieces of data.
        //1. old name, new name\n        //2. data-index
        //3. list of direct parents.\n        //4. indexes of all parents.
        var inner ={}\n        inner["new"] = newval;
        //depends if it was a span or not. Span means we use parents.
        inner["i"]= (isSpan? child.parent().attr("data-index"): 
          child.attr("data-index")) || 0\n        var allP = child.parents()
        var pNames =[]\n        var pIndex =[]
        for(var y=0; y < allP.length; y++){\n          var p = $(allP[y])
          var tag = p.prop("tagName")
          if(p.attr("data-name") == parent.attr("data-name")){
            if(addParent){\n             pNames.push(parent.attr("data-name"))
              pIndex.push(parent.attr("data-index") || 0);\n            }
            break;\n          }\n          else if(tag == "LI"){
            pNames.push(p.attr("data-name"));
            pIndex.push(p.attr("data-index") || 0);\n          }\n        }
        inner["p"] = pNames\n        inner["pIndex"] = pIndex\n       var t={}
        t[oldval]= inner\n        oldToNew.push(t)\n      }
    } // end of childchanged loop\n return oldToNew\n}
// --- end of save operations ---
// --- Expand and show more data operations ---\n/*
  Used when in edit mode to stop any items being added to the webpage.
  More items added after edit mode- they wont be editable, and when you
  click to edit something it loads its children.\n*/
$(document).on('click','.expand, .expanded', function(event){
    if(expandDisable){\n     event.stopImmediatePropagation();\n    }\n});\n/*
  Retrieves and expands elements whose objects have multiple lists
  (i.e. onese who's data-base !=simpleData)\n*/
$(document).on('click', '.Consumer.expand,.Producer.expand,'
+'.Path.expand,.EndPath.expand ',function(event){
    var allModules = CURRENT_OBJ.getModules($(this).attr('data-name'));
    var UL = addParams(this,allModules)\n    $(this).append(UL); 
  event.stopPropagation();\n});\n/*\n  Adds parameters onto objects.\n*/
$(document).on('click','.Modules.expand,.SequenceTypes,.Types.expand'
, function(event){\n  if(showParams){\n    addParams(this);\n  }
  event.stopPropagation();\n});\n/*\n  Hides/Shows children from class param.
*/  \n$(document).on('click', '.paramInner.expand, .paramInner.expanded',
   function(event){\n   if($(this).children('ul').length ==0){
     // find parents\n     var parents = findParents(this)
     var result = baseInnerParams(parents,parseInt(
                   $(this).attr("data-index")) )[1]
     addParams(this, result);\n    }\n    else{
      //children already added, so just hide/show.
      $(this).children('ul').toggle();\n    }\n    event.stopPropagation();
  });\n/*\n  Find the parents of a child.\n*/\nfunction findParents(child){
  var parents =[$(child).attr("data-name")]
  var theParent = $(child).attr("data-parent")
  while(theParent !=undefined){  \n    var child = $(child).parent();
    if(child.prop("tagName")=="UL") continue;
    parents.unshift(child.attr("data-name"));
    theParent = child.attr("data-parent");\n  }\n  return parents\n}\n/*
  Helper function: returns filename appended onto val.\n*/
function getFile(theName){\n  var f = baseFile(theName)
  if(f)return theName+=type(f)\n  return theName\n}\n/*
  Add params to the object. Object can be of any type
  (normally module or param).
  Will be used by modules adding parameters, psets adding parameters.\n*/
  var $LIBasic = $(document.createElement('li')).attr("class","param");
  var $LIExpand = $LIBasic.clone().attr("class","expand");\n
function addParams(obj, params){
  var fileChecked = document.getElementById("ShowFiles").checked
  var $span = $(document.createElement("span"));
  var $typeSpan = $span.clone().addClass("type");
  var $valSpan = $span.clone().addClass("value");
  var $UL = $(document.createElement("ul"));
  var $objName = $(obj).attr('data-name');\n\n  if(!params)
    params = baseParams($objName)\n  for(var i =0; i < params.length; i++){
    var all = params[i].slice() // make copy of it
    var isList= typeof(all)=="object"
    var theName = isList ? all.shift(): all
    var typ= !isList || !all.length ? baseType(theName): all.pop()
    var gen = getGenericType(typ) 
    var spt = $typeSpan.clone().attr("data-name",typ).text(type(typ))
    if(fileChecked) text = getFile(theName)
    if(isList && typeof(all[0]) == "object"){\n      // PSets
      var cloLI = doLI(false,theName,i,"paramInner",spt)
      cloLI.attr("data-parent", $objName)\n    }
    else if(baseParams(theName)){\n      // Modules or sequences
      var cloLI = doLI(false,theName,i,gen,spt)\n    }\n    else{
      // Basic type, has no children\n      var cloLI= doLI(true,theName,i)
      var value =""\n      if(all.length)\n        var value = all.shift()
      // formating so lots of strings look nicer\n     var valDataName = value
      if(value.indexOf(",")>-1){
         value = "<ul><li>"+value.replace(/,/g, ",</li><li>")+"</li></ul>"
      }\n      var add = type(typ)
      cloLI.append($valSpan.clone().attr("data-name",valDataName).html(value))  
      cloLI.append($typeSpan.clone().attr("data-name",add).text(add))
      for(var p=0; p < all.length; p++){\n        var n = type(all[p])       
        cloLI.append($typeSpan.clone().attr("data-name",n).text(n))\n      } 
    } \n    $UL.append(cloLI);\n  }\n  $(obj).append($UL);\n}\n
function type(theName){\n  return " ("+theName+")"\n}\n/*
  Helper function: Adds data to a LI.\n*/
function doLI(basic,dataN,dataI,classes,html){
   if(basic) var $LI = $LIBasic.clone()\n   else var $LI = $LIExpand.clone()
   $LI.attr("data-name", dataN).attr("data-index", dataI).text(dataN);
   if(classes)$LI.addClass(classes)\n   if(html)$LI.append(html)\n  return $LI
}\n/*\n  Box to show params has been clicked.\n*/
$(document).on('click', '#ShowParams', function(e){
  if($(this).is (':checked')){\n    showParams = true\n  }\n  else{
    $(this).next().hide()\n    showParams = false\n  }\n});\n/*
  Removes children from top level list elements.\n*/
$(document).on('click', '#hide', function(e){
  //make sure not called when not needed.
  if($(this).css('cursor')!='default'){  
    var selec = $(".expanded."+topClass).children("ul").hide()
    toggleExpand($(".expanded."+topClass ),e)\n    invisibleHide()\n  }\n});
// --- end of expand and show more data operations ---
// --- general helper operations and functions ---\n\n/*
  Return to normal viewing mode.\n*/\nfunction goNormal(it){
  expandDisable = false;
  $(".cellEdit").replaceWith(function() { return $(this).contents();});
  $("#save").remove()\n  $("#mode")[0].textContent = "Normal Mode"
  it.attr("value", "Edit");\n  it.attr("id", "editMode")\n}\n/*
  Set what the CURRENT_OBJ is.\n*/\nfunction setCURRENT_OBJ($element){
  var thefunction = $element.attr("data-base");
  var list = $element.attr("data-files").split(" ");\n  if(list.length >1){
    CURRENT_OBJ = window[thefunction](list[1], list[0])\n   } \n  else{
    CURRENT_OBJ = window[thefunction](list[0])\n  }\n}\n/*
  Add option in html to show/hide parameters.\n*/\nfunction paramOptions(bool){
  if(!bool){\n   $("#attachParams").empty()\n   return\n  }
  var lb= jQuery('<label/>', {\n      for:"ShowParams"\n  })
  jQuery('<input/>', {\n      type:"checkbox",\n      id:"ShowParams",
      name:"ShowParams",\n      value:"ShowParams",\n      autocomplete:"off"
    }).appendTo(lb)\n  lb.append("Show Parameters")
lb.appendTo("#attachParams")\n}\n/*\n Small info about each option.\n*/
$('#showType option').mouseover(function(){
  var docType = $(this).attr("value").toLowerCase();\n  var info;
  switch(docType){\n    case "producer":
      info="What's produced by each module."\n      break;
    case "consumer":\n      info="What's consumed by each module."
      break;\n    default:\n      info ="List of "+ docType+"s."\n  }
  $(this).attr("title", info);\n});\n/*\n Small info about each option.\n*/
$('span[name="Info"]').mouseover(function(){
  var docType = $(this).attr("value").toLowerCase();\n  var info;
  switch(docType){\n    case "producer":
      info="What's produced by each module."\n      break;
    case "consumer":\n      info="What's consumed by each module."
      break;\n    default:\n      info ="List of "+ docType+"s."\n  }
  $(this).attr("title", info);\n});\n/*\n  More info about what's shown.\n*/
$("#helpMouse").hover(function(e) {
    $($(this).data("help")).stop().show(100);
    var title = "<h6>(Read the README file!)</h6><h4>Info:</h4> "
    var expl = "<h5>Colour codes:</h5> <h6><ul><li class='Path'>pathName"+
               "     </li></ul><ul><li class='Modules'>Modules (e.g. "+
               "EDProducer, EDFilter etc)</li></ul><ul><li class='Types'>"+
               "Types (e.g. PSet)</li></ul><ul><li class='param'>"+
               "ParameterName:<span class='value'> value</span><span"+
               " class='type'>(type)</span></li></ul></h6>"
   var info ="<h5>The data</h5><h6>The headings you can choose from are"+
             " what was collected from the config file.<br/><br/> Any "+
             "change to the config file means having to run the script "+
             "again and then refresh this page (if same output file was "+
             "used).</h6><br/>"
   var tSearch="<h5>Search</h5><h6>Currently can only search by listing "+
               "what items you would like to be searched, and then what part"+
               " of each item.<br/><br/> I.e. search the producers for "+
               "certain names.</h6><br/>"
   var problems = "<h5>HTML/JSON/JS issues</h5><h6>If content isn't "+
                  "loading,or json files cannot be loaded due to browser"+
                  " security issues, try runing the local server created"+
                  " by the script. This will be in the same place as "+
                  "index.html.<br/><span class='Types'>'python cfgServer.py'"+
                  "   </span></h6><br/>"
   var editing = "<h5>Editing</h5><h6>In order to use the edit mode, you "+
                 "need to run the cfgServer.py file, this will be in the "+
                 "same directory as the index.html.<br/>Then go to <span"+
                 " class='Types'> 'http://localhost:8000/index.html.'"+
                 "</span><br/><strong>Please note that the editing of "+
                 "SequenceTypes (i.e. Paths, EndPaths etc) and of producers"+
                 " and consumers has been "
   $($(this).data("help")).html(title+expl+info+problems+editing);
                 "disabled.</strong></h6><br/>"\n}, function() {
    $($(this).data("help")).hide();\n});\n/*
  These two functions hide/show the top box in browser.\n*/
$(document).on('click', '.hideTopBox', function(e){
   $(".topSpec").animate({top: "-2.5em"}, 500);\n   $(this).text("show");
   $(this).toggleClass("hideTopBox "+"showTopBox");\n});
$(document).on('click', '.showTopBox', function(e){
   $(".topSpec").animate({top: "0em"}, 500);\n   $(this).text("hide");
   $(this).toggleClass("showTopBox "+ "hideTopBox");\n});\n/*
  Stop some links from firing.\n*/\n$('a.nowhere').click(function(e)\n{
    e.preventDefault();\n});\n\n// Turn off any action for clicking help.
$('a#help').bind('click', function() {\n   return false;\n});\n/*
  If parameter value is a list, hide/show the list on click.\n*/
$(document).on('click', '.param',function(event){\n    if(!expandDisable){
  if($(this).find('ul').length >0){\n    $(this).find('ul').toggle();\n    }}
  event.stopPropagation();\n});\n/*
  Removes children from expanded paths or modules.\n*/
$(document).on('click', '.expanded',function(event){
  var c = $(this).children('ul');\n  if(c.length >0){\n    $(c).remove();  
  }\n  event.stopPropagation();\n});\nfunction visibleHide(){
  $('#hide').css('opacity',1);\n  $('#hide').css('cursor','pointer');
  hideVisible = true;\n}\nfunction invisibleHide(){
  $('#hide').css('opacity','');\n  $('#hide').css('cursor','');
  hideVisible = false;\n}\n\n// Toggles class names.
$(document).on('click','.expand, .expanded', function(event){
  if(!hideVisible && $(this).is('.expand')){\n  visibleHide();\n  }
  toggleExpand(this, event);\n});\n/*\n  Helper function toggles class type.
*/\nfunction toggleExpand(me,event){\n    $(me).toggleClass("expanded expand");
  event.stopPropagation();\n}\n});\n// end of jquery\n/*
Function to load the JSON files.\n*/\nfunction loadJSON(theName){
 return $.ajax({\n    type: "GET",\n    url: theName,
     beforeSend: function(xhr){\n    if (xhr.overrideMimeType)\n    {
      xhr.overrideMimeType("application/json");\n    }\n  },
    contentType: "application/json",\n    async: false,\n    dataType: "json"
  });\n} \n"""%(pN,pFN))
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
#!/usr/bin/env python\nimport SimpleHTTPServer\nimport SocketServer
import shutil\nimport os\nimport cgi\nimport socket\nimport errno\n
#Right now cannot deal with SequenceTypes\nclass CfgConvert:
  def __init__(self, obj):\n    self._pName = obj["processName"]
    self._pFileN = obj["processFileName"]\n    obj.__delitem__("processName")
    obj.__delitem__("processFileName")\n    self._obj = obj
    self._func="cms.%s"
    self.header=\"""import FWCore.ParameterSet.Config as cms
process = cms.Process('%(n)s')\nfrom %(fileN)s import *\n%(changes)s\n\n\"""
  def _doConversion(self):\n    return self.header%(
    {"changes":self._doWork(self._obj),
     "n":self._pName,"fileN":self._pFileN})\n\n  def _doWork(self, obj):
    result =""\n    for key, value in obj.iteritems():
      _type = value["Type"]\n      _file = value["File"]
      _params = value["Parameters"]\n      _oType = value["oType"]
      if(_oType):\n        _oType= "'%s',"%(value["oType"])
      #f = eval(func%(_type))\n      if(type(_params)== list):
        if(len(_params)==0):
          result +="process.%s= cms.%s(%s)\\n"%(key,_type,_oType[:-1])
        else:\n          params = self._convert(_params)
          result+="process.%s= cms.%s(%s%s)\\n"%(key,_type,_oType, params)
      #elif(type(_params)== dict):
      #  return "Sorry path and endpaths can not currently be translated."
        #params = self._doWork(_params)\n      
      #obj = f("actualType", *parameters)\n    return result\n      
  def _convert(self,params):
    # NOTE: params do not take names inside functions
    # It is: name = cms.type(value)\n    # okay we have a list like 
    #[name,params, untracked,type]||[name,params,type]
    # 2 format options in the params\n    # 1. [["name", "value", "type"]]
    # 2. [["PsetName",[[as above or this]], "Pset"]]\n
    # At position 0 is name. list[0]==name or list[0]==list 
    # then that will be all
    # At position 1 is value or list. list[1]==value||list.
    # At position 2 is type. list[2] == type || untracked.
    # If there is a position 3 its type. list[3]== type.
    if(all(type(x) == list for x in params)):
      return ",".join([self._convert(x) for x in params])
    length = len(params)\n    if(length==1):
      if(type(params[0]) == list):\n        # wont have params[1]etc  
        # do inners\n        return  self._convert(params[0])
      #wrong format\n      print "Error 01 %s"%(str(params))\n      return ""
    if(length !=3 and length !=4):\n      #wrong format
      print "Error 02 len is %s"%(str(params))\n      return ""
    # okay get on with it.\n    name = params[0]
    if(type(params[1])==list):\n      # do listy things
      value = self._convert(params[1])\n    else:\n      value=params[1]
    ty = params[2]\n    if(name==ty):\n      name=""\n    if(length==4):
      ty+="."+params[3]\n      if(name== params[3]):\n        name=""
    if("vstring" in ty):\n      # split the value up by commas
      value = ",".join(["'%s'"%x for x in value.split(",")])
    elif("string" in ty):\n      #surround string with quotation marks
      value = "%s"%(value)\n    # okay now done with everything
    call= self._func%(ty)\n    if(name):
      return"%s=%s(%s)"%(name,call,value)\n    return"%s(%s)"%(call,value)\n
class ServerHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):\n
  def do_GET(self):\n    if(self.path == "/index.html"):
      # find out what cfg html folders we have below
      # and add each main html page to the index\n      li = os.listdir(".")
      dEnd = "-cfghtml"
      dirs = [x for x in li if os.path.isdir(x) and x.endswith(dEnd)]
      name = "main.html"
      names = [os.path.join(x,name) for x in dirs if name in os.listdir(x)]
      tmpte = '<li><a href ="%(n)s">%(s)s</a></li>'
      lis = [tmpte%{"n":x,"s":os.path.split(x)[0].replace(dEnd, "")}
               for x in names]\n      with open("index.html", 'w') as f:
        f.write(\"""\n<!DOCTYPE html>\n<html>\n  <head>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8">
    <title>cfg-browser</title>\n  </head>\n  <body>
    <h4>Configuration files:</h4>\n    <ul>\n      %s\n    </ul>\n  </body>
</html>\n        \"""%("".join(lis)))\n      
    return SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)\n  
  def do_POST(self):
    ctype,pdict= cgi.parse_header(self.headers.getheader('content-type'))
    bdy = cgi.parse_multipart(self.rfile,pdict)
    ch= " ".join(bdy[bdy.keys()[0]])\n    self.conversion(eval(ch))
    self.writeFile()\n    print "finished writing"\n    
  def conversion(self, json):\n    # Need to convert my json into config
    result = CfgConvert(json)._doConversion()
    with open("changed_cfg.py", 'w')as f:\n      f.write(result)\n
  def writeFile(self):\n    with open("changed_cfg.py", 'rb') as f:
      self.send_response(200)
      self.send_header("Content-Type", 'text/html')
      self.send_header("Content-Disposition", 
                       'attachment; filename="changed_cfg.py"')
      fs = os.fstat(f.fileno())
      self.send_header("Content-Length", str(fs.st_size))
      self.end_headers()\n      shutil.copyfileobj(f, self.wfile)\n
def main(port=8000, reattempts=5):\n  if(port <1024):
    print \"""This port number may be refused permission.
It is better to use a port number > 1023.\"""\n  try:
    Handler = ServerHandler
    httpd = SocketServer.TCPServer(("", port), Handler)
    print "using port", port
    print "Open http://localhost:%s/index.html in your browser."%(port)
    httpd.serve_forever()\n  except socket.error as e:
    if(reattempts >0 and e[0] == errno.EACCES):
      print "Permission was denied."
    elif(reattempts >0 and e[0]== errno.EADDRINUSE):
      print "Address %s is in use."%(port)\n    else:\n      raise
    print "Trying again with a new port number."\n    if(port < 1024):
      port = 1024\n    else:\n      port = port +1
    main(port, reattempts-1)\n\nif __name__ == "__main__":\n  import sys
  if(len(sys.argv)>1):\n    try:\n      port = int(sys.argv[1])
      main(port)\n      sys.exit() \n    except ValueError:
      print "Integer not valid, using default port number."\n  main()
""")

def computeConfigs(args):
  pyList=[]
  for x in args:
    if(os.path.isdir(x)):
      # get all .py files.
      allItems = os.listdir(x)
      py = [os.path.join(x,y) for y in allItems if y.endswith(".py")]
      pyList.extend(computeConfigs(py))
      if(recurse):
       # print "recurse"
        # if we want to recurse, we look for everything
        dirs = []
        for y in os.listdir(x):
          path = os.path.join(x,y)
          if(os.path.isdir(path)):
            pyList.extend(computeConfigs([os.path.join(path,z)
                                          for z in os.listdir(path)]))
    elif(x.endswith(".py")):
      pyList.append(x)
  return pyList
 
recurse=False

def main(args,helperDir,htmlFile,quiet, noServer):
  dirName = "%s-cfghtml"
  # new dir format
  #  cfgViewer.html
  #  patTuple-html/
  #    lower.html
  #    cfgViewerHelper/
  #      -json files
  pyconfigs = computeConfigs(args) 
  tmpte = '<li><a href ="%(n)s">%(s)s</a></li>'
  lis=""
  found =0
  for x in pyconfigs:
    print "-----"
    # for every config file
    name = os.path.split(x)[1].replace(".py", "")
    dirN =  dirName%(name)
    # we have the dir name now we only need
    # now we have thedir for everything to be stored in
    #htmlF = opts._htmlfile
    #htmldir= os.path.split(htmlFile)[0]
    #baseDir = os.path.join(htmldir,dirN)
    baseDir = dirN
    dirCreated = False
    if not os.path.exists(baseDir):
      os.makedirs(baseDir)
      dirCreated = True
    # base Dir under where the htmlFile will be.
    lowerHTML = os.path.join(baseDir, htmlFile)
    helper = os.path.join(helperDir, "")
    helperdir = os.path.join(baseDir, helper, "")
    if not os.path.exists(helperdir):
      os.makedirs(helperdir)
    print "Calculating", x
    try:
      u = unscheduled(x, lowerHTML, quiet, helper,helperdir)
    except Exception as e:
      print "File %s is a config file but something went wrong"%(x)
      print "%s"%(e)
      continue
    print "Finished with", x
    if(not u._computed and dirCreated):
      # remove any directories created
      shutil.rmtree(baseDir)
      continue
    found +=1
    lis += tmpte%{"n":os.path.join(dirN,htmlFile),"s":name}
  with open("index.html", 'w')as f:
    f.write("""
<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8">
    <title>cfg-browser</title>
  </head>
  <body>
    <h4>Configuration files:</h4>
    <ul>
      %s
    </ul>
  </body>
</html>
    """%("".join(lis)))
  if(found == 0):
    print "Sorry, no configuration files were found."
    return
  print "Finished dealing with configuration files."
  server("cfgServer.py")
  if(not noServer):
    print "-----"
    print "Starting the python server.."
    import  cfgServer
    cfgServer.main()


if __name__ == "__main__":
  import sys, os, imp, shutil
  from optparse import OptionParser
  parser = OptionParser(usage="%prog <cfg-file> ")
  parser.add_option("-q", "--quiet",
                  action="store_true", dest="_quiet", default=False,
                  help="Print minimal messages to stdout")
  #parser.add_option("-o", "--html_file", dest="_htmlfile",
  #                  help="The output html file.", default="cfg-viewer.html")
  parser.add_option("-s", "--no_server",
                  action="store_true", dest="_server", default=False,
                  help="Disable starting a python server to view "\
                  "the html after finishing with config files.")
  parser.add_option("-r", "--recurse", dest="_recurse",action="store_true",
                     default=False,
                     help="Search directories recursively for .py files.")
  # store in another dir down
  #additonalDir = "first"
  helper_dir = "cfgViewerJS"
  opts, args = parser.parse_args()
  #cfg = args
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
  recurse = opts._recurse
  doesNotExist = [x for x in args if not os.path.exists(x)]
  if(len(doesNotExist)):
    s =""
    for x in doesNotExist:
      args.remove(x)
      s += "%s does not exist.\n"%(x)
    print s
    
  if(len(args)==0 or len(doesNotExist)== len(args)):
    print """
    Either you provided no arguments, or the arguments provided do not exist.
    Please try again.
    If you need help finding files, provide arguments "." -r and I will search
    recursively through the directories in the current directory.
    """
  else:
    main(args,"cfgViewerJS","main.html",opts._quiet, opts._server)
  
