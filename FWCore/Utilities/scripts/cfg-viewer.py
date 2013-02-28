#!/usr/bin/env python
from optparse import OptionParser
import imp
import re
import sys
import shutil
import os
import FWCore.ParameterSet.Config as cms
    
modsNames=[]
nameToFile= {}
newModsNames,oldMods,allMods=[],[],[]

class generateBrowser:
  def __init__(self,htmlFile, cfgFile):
    self.html = htmlFile
    self.theDir=''
    self.pathName = "paths.html"
    self.js = "search.js"
    self.thecss = "style.css"

    rest= html.rsplit('/',1)    
    # Find where the html file is to be stored.
    # store everything else there, move other stuff there.
    if(len(rest)>1):
      self.theDir=rest[0]+'/'
      if(not os.path.isdir(self.theDir)):
        os.mkdir(self.theDir)
      for e in ["More.png","Less.png","jquery-1.8.3.min.js"]:
        shutil.copyfile(e,self.theDir+e)
    self.javascript("%s%s"%(self.theDir,self.js))
    self.css("%s%s"%(self.theDir,self.thecss))
    self.pathLi="""
    <li class="expand %(className)s" data-name="%(nameID)s">%(nameID)s:</li>
    """ 
    self.cfg = imp.load_source("cfg", cfgFile)
    # do the psets (in other file, provide link to it.)
    psetTypes = open("%sparamTypes.js"%(self.theDir),'w')
    psetTypes.close()
    print "Starting print out of psets."
    self.doPsets()
    print "Finished finding psets. Now onto paths..."
    self.doPaths()
    print "Paths done"

  def doPaths(self):
    modHTMLName= "%sparamTypes.js"%(self.theDir)
    modToPathName = "%smodToPath.js" %(self.theDir)
    tempFile ="%stemp.js"%(self.theDir)
    f={modHTMLName:"params",modToPathName:"modules"}
    o={modHTMLName:"a",modToPathName:"w"}
    theS="var %s={"
    for key in f.keys():
      curFile = open(key,o[key])
      curFile.write(theS%(f[key]))
      curFile.close() 
    first = True
    liToBeWritten, frmPathHolder="", "";
    className, theS,frmPathStr ="path","",'%s%s:["%s"]'
    newModsStr, oldModsStr='%s\n%s:["%s"]', '%s"%s",%s'
    dataFile= open(modHTMLName,'a')
    v = visitor(dataFile)
    paths = self.cfg.process.paths
    global oldMods
    global allMods
    global newModsNames
    for item in paths.keys():
      oldMods=[] # module names which have been seen before.
       # needed so we can add this path name to the dictionary in modToPath.js
      allMods =[]# set of current path modules names including duplciates.       
      newModsNames=[] # set of current paths module name without duplicates.
      # could get oldMods from all Mods if we take away all newModsNames
      liToBeWritten +=format(self.pathLi,nameID=item,className=className)
      paths[item].visit(v)
      frmPathHolder +=frmPathStr%(theS,item,"\",\"".join(allMods))
      fromMod = open(modToPathName, 'r')
      fromModTemp = open(tempFile, 'w')
      for line in fromMod:
        tLine= re.sub('\[.*?\]', "",line) 
        result = re.split(':|,', tLine[tLine.find('{')+1:])
        namesFound = [x for i, x in enumerate(result)if not i%2]
        # the new names we have is going to be smaller, so we'll loop round that
        for each in oldMods:
          if(each in namesFound):
            oldMods.remove(each)
            index = line.find(each)+len(each)+2 #  as will have :[ after it
            line =  oldModsStr %(line[:index],item,line[index:])
        fromModTemp.write(line)
      fromMod.close()
      for each in newModsNames:
        fromModTemp.write(newModsStr%(theS,each,item))
        theS=","
      fromModTemp.close()
      os.rename(tempFile,modToPathName)
      if(first):
        theS=","
        first=False
    out = open("%s%s"%(self.theDir,self.pathName), 'w')
    self.printStart(self.js,self.thecss, """disabled="True" """, out)
    out.write("<a href='%s'> See Psets</a>&nbsp;\n<ul>%s"\
      "</ul>\n</body>\n</html>"%(self.html.rsplit('/',1)[-1],liToBeWritten))
    out.close()
    dataFile.write("};\nfunction getAllParams(){\n"\
                    "return params;}\n"\
                    " function getParams(modName){\n"\
                    " return params[modName];}\n"\
                    "function getInnerParams(parentsNames){\n"\
                    "var currentList = params[parentsNames[0]];\n"\
                    "for(var i=1; i < parentsNames.length;i++){\n"\
                    "  for(var p=0; p < currentList.length;p++){\n"\
                    "      if(currentList[p][0]==parentsNames[i]){\n"\
                    "        var found = currentList[p][1];\n"\
                    "        break;\n }\n}\n currentList = found;\n}\n"\
                    "  return currentList;}")
    fromPath= open("%spathToMod.js" %(self.theDir),'w')
    fromPath.write("var paths={%s}\n "\
     "function getModules(thePath){\n"\
    "return paths[thePath];\n}"\
    "\nvar filenames=%s\n function"\
    " getFile(name){\n return filenames[name]; \n}"\
    "\n function keys(){\n return Object.keys(paths); "\
    "\n}" %(frmPathHolder,str(nameToFile)))
    fromPath.close()
    fromMod = open(modToPathName, 'a')
    fromMod.write("}\n function getPaths(theMod){\n"\
    "return modules[theMod];\n}")
    fromMod.close()

  def doPsets(self):
    addC,writeOut,toParamWrite= "","",""
    first = True
    outStr=self.pathLi
    pStr="%s%s:'%s'"
    classN = "module"
    process = self.cfg.process
    psets = process.psets
    theDict ={}
    for pset in psets:
      writeOut+=format(outStr,nameID=pset,className=classN)
      psetCfg = process.__dict__[pset]
      res = do(psetCfg.parameters_(),[])
      theDict[pset] = res
      if(first):
        addC=","
        first=False
    out = open(self.html, 'w')
    self.printStart(self.js, self.thecss ,"", out)
    out.write("<a href=\"%s\"> See Paths</a>\n<ul>"\
            "%s</ul></html>"%(self.pathName,writeOut))
    out.close()

    psetTypes = open("%sparamTypes.js"%(self.theDir),'w')
    psetTypes.write("psetParams=%s\nfunction getAllpsetParams(){\n"\
                    "return psetParams;}\n"\
                    " function getpsetParams(modName){\n"\
                    " return psetParams[modName];}\n"\
                    "function getInnerpsetParams(parentsNames){\n"\
                    "var currentList = psetParams[parentsNames[0]];\n"\
                    "for(var i=1; i < parentsNames.length;i++){\n"\
                    "  for(var p=0; p < currentList.length;p++){\n"\
                    "      if(currentList[p][0]==parentsNames[i]){\n"\
                    "        var found = currentList[p][1];\n"\
                    "        break;\n }\n}\n currentList = found;\n}\n"\
                    "  return currentList;}"%(theDict))

  # Start of html files.
  def printStart(self,js, css, dis, out):
    scripts ="""
   <script type="text/javascript" src="paramTypes.js"></script>
    """
    classType="pset"
    buttons="""
    <option id="module" selected="selected">Module</option>
    """
    if(len(dis)>0):
      classType="path"
      buttons="""
      <option id="path" selected="selected">Path</option>
      <option id="module">Module</option>
      <option value="type">Type</option>
      <option value="parameter">Parameter</option>
      <option value="value">Value</option>
      """
      scripts= """
      <script type="text/javascript" src="pathToMod.js"></script>
      <script type="text/javascript" src="modToPath.js"></script>
      <script type="text/javascript" src="paramTypes.js"></script>
      """
    out.write( """
    <!DOCTYPE html>
    <html>
      <head>
        <title>cfg-browser</title>
        <script src="jquery-1.8.3.min.js" 
        type="text/javascript"></script>
        %(scripts)s
        <script type="text/javascript" src="%(js)s"></script>

        <link href="%(css)s" rel="stylesheet" type="text/css"/>
      </head>
      <body>
      <!--form name="input" action="html_form_action.asp" method="get"--> 
      <form name="searchInput" onsubmit="return false;">
      <input type="text" id="searchWord"/>
      <select id="searchType">
      %(option)s
      </select>
       
       <input type="submit" value="Search" id="search" disabled/></form> 
      <br/><p id="searchNumber"></p><br/>
      <a id="hide" style="cursor:default; color:#CCCCCC;" 
      class="%(class)sReset"
      href="javascript:;" >
       Hide All</a>
      <br/><br/>
   <span id="pageType" data-type=%(class)s></span>

  <input type="submit" id ="searchReset" style="display:none" 
  value="Reset search results." class="%(class)sReset"/>

    """%{'js':js,'option':buttons, 'css':css, 'dis':dis,
       'scripts':scripts, 'class':classType})

  def javascript(self,thejs):
    jsFile = open(thejs, 'w')
    jsFile.write( """    
    
     
       
     
   $(document).ready(function(){ 
$(document).on('click', '#search', function(e){
     
    numFound = 0;
    var par = $(this).parent();
    var id = $(par).find('#searchType').attr('id');
    var option = $('#'+id+' option:selected').text();
    var val = $(par).find('#searchWord').val()
    var reg = new RegExp("("+val+")", "gi");
    switch(option){
    case "Path":
      reset('li.path', e, true);
      numFound = topLevelMatch(keys(),reg,false); // will stay the same
    
    case "Module":
      if(typeof modKeys != 'function' && false){
        //if modKeys is not defined means we're using pset html.
        //maybe change this so instead we define top layer li with class top
        reset('li.module', e, true);
        //numFound = topLevelMatch
        numFound = topLevelMatch( $('li.module'),reg,true);
      }
      else{
       // for now pretend that its in html.
        //searchReplaceHTML(reg, "module");
        // if not in the html
        //Okay can either, get allmatched modules and add them to all html parents
        // or go round paths and get all matched modules adding to tghe path
        var strings = allMods(reg);
        var Li = $(document.createElement('li')).attr("class","module expand");
        console.log("hereiam before");
        searchReplaceNonHTML(strings,Li, reg)
      } //else
     //elseif
   
    case "File Name":
      //get all modules names, get files for all modules
      // once we have a list of all modules we want
      // add those modules to
    
    case "Parameter":
      // gives the params which match the reg.
      numFound = lowerLevels(reg,doInnerParams, e)
      
    case "Type":
      reset('li.path',e,true);
      // similar to params except we're looking at the types and not the params names.
      // so need to get all the params, loop over the keys
      var matches = matchValue(reg);
      console.log("matches are "+ matches)
    
    case "Value":
      numFound = lowerLevels(reg,doInnerValue, e)
    }
    $('#searchNumber').html(numFound+" found.");
    $('#searchReset').show();
  });

//do Params, values.
function lowerLevels(reg,funcCall, e){
  reset('li.path', e, true);
  var numFound=0;
  
   if($("span[id='pageType']").attr("data-type")=="pset"){
        var matches = getAllpsetParams();
      }
      else var matches = getAllParams();;

  var keys = Object.keys(matches);
  for(var i=0; i < keys.length; i++){
    var theKey = keys[i];
    var params = matches[theKey];
    var theMod = moduleLI.clone().attr("data-name",theKey).text(theKey);
    // need to make the ul 
    tempNumber =0;
    var theUL = funcCall(params,reg);
    if(tempNumber ==0)continue;
    theMod.append(theUL);
    // for now just do parents
    var pathsAddTo = getPaths(theKey);
    var mul =0;
    for(var k=0; k < pathsAddTo.length;k++){
      // all paths to be added to.
      var theP = pathsAddTo[k];
      // find the parent.
      $('li.path[name='+theP+']').each(function(){
        // for each one add the theMod.
        $(this).children().empty();
        $(this).append(theMod.clone());
        mul +=1;
       });
    }
    numFound += (tempNumber*mul)
  }
  return numFound;
}

function doInnerValue(modules,reg){
  var theUL = UL.clone();
  // okay we have a list where we want to check the inner list is an inner list.
  for(var i=0; i < modules.length; i++){
    var innerList= modules[i];
    var theName = innerList[0];
    var next = innerList[1];
    // next could be inner or could be what we want.
    if(typeof(next)=="object"){
      // we have inner params, recurse!
      var oldNum = tempNumber 
      var newUL = doInnerValue(next, reg); // will return a ul.
      if(tempNumber == oldNum){
        var li= LIExpand.clone().attr("data-name", theName).html(theName);
      }
      else {
        var li= LIExpanded.clone().attr("data-name", theName).html(theName);
        li.append(newUL);
      }
    }
    else{
      // else do normal and make the
      // rename the value
      var newValue = next.replace(reg, "<em>$1</em>"); 
      if(newValue != next){
         tempNumber +=next.match(reg).length;
      }   
      var li= paramLI.clone().attr("data-name",theName).html(theName);
      var theClass="value";
      li.append(span.clone().attr("class", "value").html(": "+newValue));
      innerList.slice(2).forEach(function(i){
        li.append(span.clone().attr("class", "type").text(" ("+i+")"));
      });
    }
    theUL.append(li);
  }
  return theUL;
  }

function doInnerParams(params,reg){
  var theUL = UL.clone();
  // okay we have a list where we want to check the inner list is an inner list.
  for(var i=0; i < params.length; i++){
    var param = params[i];// it's name.
    //console.log("param is "+ param)
    var theName = param[0];
    //console.log("theName "+ theName);
    var newName = theName.replace(reg, "<em>$1</em>");
    if(newName != theName){
       tempNumber +=theName.match(reg).length;
    }
    var next = param[1];
    //console.log("next is "+ next);
    if(typeof(next)=="object"){
      // we have inner params, recurse!
      //var cloLI= LIExpand.clone().attr("name", theName).text(theName);
      //var ul = UL.clone();
      var old = tempNumber;
      var newUL = doInnerParams(next, reg); // will return a ul.
      if(old == tempNumber){
        var li= LIExpand.clone().attr("data-name", theName).html(newName);
      }
      else {
        console.log("newName is "+ newName);
        var li= LIExpanded.clone().attr("data-name", theName).html(newName);
        li.append(newUL);
      }
    }
    else{
      // else do normal and make the 
      var li= paramLI.clone().attr("data-name",theName).html(newName);
      var theClass="value";
      for(var w=1; w < param.length;w++){
        // do types etc.
        if(w==1)li.append(span.clone().attr("class", "value").text(": "+param[w]));
        else  li.append(span.clone().attr("class", "type").text(" ("+param[w]+")"));
      }
    }
    theUL.append(li);
  }
  return theUL;
  }


  var LI= $(document.createElement('li'));
  var moduleLI= LI.clone().attr("class","module expand");
  var paramLI= LI.clone().attr("class","param expanded");
  var UL = $(document.createElement('ul'));
  var span = $(document.createElement('span'));
  var LIExpand = LI.clone().attr("class","expand paramInner");
  var LIExpanded = LI.clone().attr("class","expanded paramInner");
// All params send here.
//[[normaloparam, value,type][inner[innervalues, value,type]][norm, value, type]]
// we will get [normaloparam, value,type] etc
var haveFoundInInner = false;
var numFound =0;
var tempNumber =0;


/*
  Added diff is that what we have is not in html yet.
  Options:
    we get the strings, if matched we change them,
    find what there parents are add to parents, show all parents so
    strings = find(fromThisGetStrings)
    for str in strings:
      if(str.match(find)){
        
      }
*/
  /*
   String format should have where they want the string to go as name.
   For module just now. //
  */// stringformat would be what we want it to go in,
  // Okay so for module i should have <li class='module expand' name='thename'> thename</li>
  function searchReplaceNonHTML(strings, theLi, regex){
    // we have the things we want, the string should just be the
    // what we want to be shwn i.e. an LI, which we clone.
    var UL = $(document.createElement('ul'));
    for(var i=0; i <strings.length; i++){
      var thisOne = strings[i]; // e.g. generator
      //here i have the string and now i will
      //need a format string or something, to know what to put around it?
      var LI = theLi.clone().attr("data-name", thisOne).html(thisOne.replace(regex, "<em>$1</em>"));
      var paths = getPaths(thisOne);
      for (var p=0; p < paths.length; p++){
        // need to find the path on the page and add the module to it.
        $('li.path[name='+paths[p]+']').each(function(){
          // for each add this LI
          if($(this).children('ul').length ==0){
            $(this).append(UL.clone().append(LI.clone()));
          }
          else{
            $(this).children('ul').append(LI.clone()); // so all paths dont point to the same LI,maybe can change this? 
          }
          $(this).attr("class","expanded path");
        });
      }
    }
  }

  function searchReplaceNonHTMLParams(strings, theLi, regex){
    // we have the things we want, the string should just be the
    // what we want to be shwn i.e. an LI, which we clone.
    var UL = $(document.createElement('ul'));
    for(var i=0; i <strings.length; i++){
      var thisOne = strings[i]; // e.g. generator
      //here i have the string and now i will
      //need a format string or something, to know what to put around it?
      var LI = theLi.clone().attr("data-name", thisOne).html(thisOne.replace(regex, "<em>$1</em>"));
      getModules(regex);
      var paths = getPaths(thisOne);
      for (var p=0; p < paths.length; p++){
        // need to find the path on the page and add the module to it.
        $('li.path[name='+paths[p]+']').each(function(){
          // for each add this LI
          if($(this).children('ul').length ==0){
            $(this).append(UL.clone().append(LI.clone()));
          }
          else{
            $(this).children('ul').append(LI.clone()); // so all paths dont point to the same LI,maybe can change this? 
          }
          $(this).attr("class","expanded path");
        });
      }
    }
  }


// search should be
/*
  identifer can be  = ["module", "type" etc]
  find = "regex"
$("li .identifer").each(function(){
  if(var howmany = $(this).attr("name").match(find)){
   // we found you 
   found +=howMany
   .replace(find, <em>+find+<em>)
  // now find parents
  // if parents not already showen show them
  $(this).parents('ul').each(function(){
    if(this not in foundParents)$(this).show()
    else{
     // already done these parents
     return false (i.e. break out loop);
    }

})

}

});

*/
  /*
   Search when what we're looking for is in the HTML but not topLevel element.
   To be global replace, find should be regexp object with g.
  */
  function searchReplaceHTML(find,identifier){
    var found =0;
    var foundParents=[];
    var notFoundLI=[];
    $("li ."+identifier).each(function(){
      var howMany=0;
      if(howmany = $(this).attr("data-name").match(find)){
       // we found you 
       found +=howMany
       $(this).html() = $(this).html().replace(find,"<em>$1</em>");
      // now find parents
      // if parents not already showen show them
      $(this).parents('ul').each(function(){
        if(foundParents.indexOf(this) ==-1){
          $(this).show();
          foundParents.append(this);
        }
        else{
         // already done these parents
         return false;// (i.e. break out loop);// the parents loop.
        }
      });
    }
    else{
     notFoundLI.append(this);
     // not a match, can just try and find a ul parent, if there is one, then we need to know whether to
     // hide it.TODO or do nothing.
    }
    });
    // now we have list of parents we want to hide all other parents
    $(notFoundLI).each(function(){
      $(this).parents('ul').each(function(){
        if(foundParents.indexOf(this) ==-1){
          $(this).hide();
        }
        else{
         // already done these parents
         return false;// (i.e. break out loop);// the parents loop.
        }
      });
    

    });
    return found;
  }
  /*
    Resets search ouput.
  */
  $(document).on('click', '#searchReset', function(e){
    $('#searchNumber').html("");
    if($(this).attr('class') == "pathReset"){
      reset('li.path',e,true); 
    }
    else{
      reset('li.module',e,true); 
    }
    $(this).hide();
  });
  
  /*
    Hides children of top level. TODO: Done - works
  */
  $(document).on('click', '#hide', function(e){
    //for hiding we just get top level and remove children.
    if($(this).css('cursor')!='default'){
      $('#searchNumber').html("");
      if($(this).attr('class') == "pathReset"){
        var selec ='li.path';
      }
      else var selec ='li.module';
      $(selec).each(function(){
        if(removeChildren(this, e))toggleExpand(this, e);
      });
      $(this).css('color','#CCCCCC');
      $(this).css('cursor','default');
    }
  });

  /*
    Retrieves and expands path elements. - works
  */
  $(document).on('click', '.path.expand',function(event){
    var UL = $(document.createElement("ul"));
    var LI = $(document.createElement('li')).attr("class","module expand");
    // temp change - name to value
    console.log($(this).attr('data-name'))
    var results = getModules($(this).attr('data-name'));
    for(var i=0; i < results.length; i++){
      var theName = results[i];
      var val = theName+" ("+getFile(theName)+")"
      UL.append(LI.clone().attr("data-name", theName).text(val));
    }
    $(this).append(UL); 
    $('#hide').css('color','')
    $('#hide').css('cursor','')
  });
  
  /*
    Retrieve and expands module elements.
    //changed to deal with new data format, (data now seperate from html specification) - works
  */
  $(document).on('click','.module.expand', function(event){
    addParams(this);
    event.stopPropagation();
  });

  /*
    Add params to the object. Object can be of any type (normally module or param).
    It's name needs to be in the data to find its parameters. - works
  */
  function addParams(obj, results){
    var LIBasic = $(document.createElement('li')).attr("class","param");
    var LIExpand = LIBasic.clone().attr("class","expand paramInner");
    var span = $(document.createElement("span"));
    var UL = $(document.createElement("ul"));
    // getParams returns list of list.
    // Format:[[name,value,type,trackedornot]].CHANGED!!
    // new format :[[name, value,type,trackedornot],[namePset,[name,value,type]]]
    // If 2nd element is list then its a pset.
    //if(!results)var results = getParams($(obj).attr('name'));
    if(!results){
      if($("span[id='pageType']").attr("data-type")=="pset"){
        var results = getpsetParams($(obj).attr('data-name'));
      }
      else var results = getParams($(obj).attr('data-name'));
    }
    for(var i =0; i < results.length; i++){
      var all = results[i].slice();
      var theName = all.shift(); 
      if(typeof(all[0]) == "object"){
        var cloLI= LIExpand.clone().attr("data-name", theName).text(theName);
        //for(var p=0; p < all[0].length;p++){
          // add all children.
        //}
      }
      else{
        // Not a Pset.
        var cloLI = LIBasic.clone().attr("data-name", theName).text(theName);
        cloLI.append(span.clone().attr("class","value").text(": "+all.shift()))  
        for(var p=0; p < all.length; p++){       
          cloLI.append(span.clone().attr("class","type").text(" ("+all[p]+")"))
        } 
      } 
      UL.append(cloLI);
    }
    $(obj).append(UL);
    $('#hide').css('color','')
    $('#hide').css('cursor','')
  }
  /*
    Hides/Shows children from class param.
    //changed to deal with new data format - works
  */  
  $(document).on('click', '.param.expand, .param.expanded',function(event){
    if($(this).children('ul').length ==0){
      addParams(this);
    }
    else{
      $(this).children('ul').toggle();
    }
    event.stopPropagation();
  });

  /*
    Hides/Shows children from class param.
    //changed to deal with new data format - works TODO add in colour scheme for innerParams
  */  
  $(document).on('click', '.paramInner.expand, .paramInner.expanded',function(event){
   if($(this).children('ul').length ==0){
     // find parents
     var theClass =""
     var obj = this;
     var parents =[$(this).attr("data-name")]
     while(theClass.indexOf("module")==-1){
       obj = $(obj).parent();
       if(obj.prop("tagName")=="UL") continue;
       var parName = obj.attr("data-name");
       parents.unshift(parName);
       theClass = obj.attr("class");
      }
      if($("span[id='pageType']").attr("data-type")=="pset"){
        var result = getInnerpsetParams(parents);
      }
      else var result = getInnerParams(parents);
      addParams(this, result);
    }
    else{
      $(this).children('ul').toggle();
    }
    event.stopPropagation();
  });

  // Needed to stop the propagation of event when it should not be expanded.
  $(document).on('click', '.param',function(event){
    event.stopPropagation();
  });

  /*
    Removes children from expanded paths or modules.
  */
  $(document).on('click', '.path.expanded, .module.expanded',function(event){
    removeChildren(this,event);
  });
  
  // Toggles class names.
  $('.expand, .expanded').live("click", function(event){
    toggleExpand(this, event);
  });
  
  //From here javascript helper functions.
  
  /*
    Does matching for top level list elements.
    Returns number of matches.
  */
  function topLevelMatch(list, reg, haveLIs){
    var numFound =0;
    for(var p=0; p < list.length; p++){
      if(haveLIs){
       var li = $(list[p]);
       var item = li.attr('data-name'); 
      }
      else{
        var item = list[p];
        var li= $('li[name='+item+']');
      }
      if(num=item.match(reg)){
        numFound +=num.length;
        li.html(li.html().replace(reg, "<em>$1</em>"));
      }
      else{
        li.hide();
      }
    }
    return numFound;
  }
  
  /*
    Removes highlights from selector, removes also children 
    if rmChildren == true.
  */
  function reset(selector, e, rmChildren){
    console.log("remove children "+ rmChildren);
    var rm = new RegExp('<em>|</em>', 'g');
    $(selector).each(function(i){
    var html = $(this).html();
    if(html.match(rm)){
      $(this).html(html.replace(rm, '')); 
    }
    else{
      $(this).show(); 
    }
    if(rmChildren)if(removeChildren(this, e))toggleExpand(this, e);}); 
  }
  
 
  /*
    Removes children from parent.
  */
  function removeChildren(parent, event){
    var c = $(parent).children('ul');
    if(c.length >0){
      $(c).remove();
      event.stopPropagation();
      return true;
    }
    return false;
  }
   /*
    Helper function toggles classes.
  */
  function toggleExpand(me,event){
    $(me).toggleClass("expanded expand");
    event.stopPropagation();
  }
});

    """)
    jsFile.close()

  def css(self,thecss):
    cssFile= open(thecss, 'w')
    cssFile.write( """
em {
  background-color: rgb(255,255,0);
  font-style: normal;
}
.module{
  color: #0000CC
}
.param {
  color: #9999CC;
  cursor:default; 
}
.value{
   color:#0000FF;
}
.type{
  color: #00CCFF;
}

ul {
  list-style-type:none;
  padding-left:0.6em;
}
li {
  background-repeat: no-repeat;
  padding-left: 1.5em;
}

.expanded, .param {
  background-image: url(Less.png);
}
.expand {
  background-image: url(More.png);
}
.expand, .expanded{
  cursor:pointer;
}
  
    """)
    cssFile.close()

"""
  Do Module Objects e.g. producers etc
"""
def doModules(modObj, i, dataFile):
  name = modObj.label_()
  allMods.append(name)
  if(name not in modsNames):
    theList = do(modObj.parameters_(), [])
    modsNames.append(name)
    newModsNames.append(name)
    nameToFile[name] = modObj._filename.split("/")[-1]
    theS =""
    if(len(modsNames) > 1): theS=","
    theS+="%s:%s"
    dataFile.write(theS%(name, theList)) 
  else:
    oldMods.append(name) 

def format(s, **kwds):
  return s % kwds
  
class visitor:
  def __init__(self, df):
    self.number = 0
    self.df = df

  def enter(self, value):
    if(isinstance(value,cms._Module)):
      doModules(value, str(self.number), self.df)
      self.number +=1

  def leave(self, value):
    num=0   

"""
 Prints out inner details of parameters.
"""
def do(params, o):
  for item in params:
    thing = params[item]
    if(hasattr(thing, "parameters_")):
      theList =[]
      theList.append("%s(%s)"%(item,thing.configTypeName()))
      # do will now return the list that the thing takes
      theList.append(do(getattr(thing,"parameters_")(),[]))
      o.append(theList)
    elif(thing.configTypeName()== "VPSet"):
          theList =[]
          theList.append("%s(%s)"%(item,thing.configTypeName()))
          newInS = "%s-%d"
          li2 =[]
          for popped in thing:
            if(hasattr(popped, "parameters_")):
              innerList =[]
              innerList.append("(%s)"%(popped.configTypeName()))
              innerList.append(do(getattr(popped,"parameters_")(),[]))
              li2.append(innerList)
          theList.append(li2)   
          o.append(theList)
    else:
      # easy version. Just save what we have - all going to have the same module..
      # have parent as an input, so we can send it.
      theList =[]
      theList.append(item)
      if(hasattr(thing, "pop") and len(thing)>0):
        value = "%s"% (",".join(str(li) for li in thing))
      else:
        value = thing.configValue()
      theList.append(value)
      theType = thing.configTypeName()
      if(theType =="double" or theType =="int"):
        theList.append(theType)
      else:
        if(thing.isTracked()):
          theList.append(theType)
          theList.append("tracked")
      o.append(theList)        
  return o

if __name__ == "__main__":
  parser = OptionParser(usage="%prog <cfg-file> <html-file>")
  flags =['-c', '-o']
  parser.add_option("-c", "--cfg_file", dest="_cfgfile",
                    help="The configuration file.")
  parser.add_option("-o", "--html_file", dest="_htmlfile",
                    help="The output html file.")
  opts, args = parser.parse_args()
  cfg, html = opts._cfgfile, opts._htmlfile
  more =0
  cfgGet, htmlGet = False, False
  if(cfg == None):
     cfgGet = True
     more +=1
  if(html == None):
    htmlGet = True
    more+=1 
  if len(args) < more:
    parser.error("Please provide one and only one configuration"\
                  "file and one output file.")
  if(cfgGet): cfg = args[0]
  if(htmlGet): html = args[1]
  try:
    f = open(cfg)
  except:
    parser.error("File %s does not exist." % cfg) 
  generateBrowser(html,cfg)
