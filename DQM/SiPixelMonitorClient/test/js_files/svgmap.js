 var SvgMap = {} ;

 SvgMap.thisFile	= ".SvgMap.js" ;
 SvgMap.theZoomAmount   = 1.05 ;
 SvgMap.theStepAmount   = 25 ;
 SvgMap.zoomAmount	= SvgMap.theZoomAmount ;
 SvgMap.stepAmount	= SvgMap.theStepAmount ;
 SvgMap.theViewText	= null ;
 SvgMap.theElementText  = null ;
 SvgMap.theSelectedText = null ;
 SvgMap.theClipArea     = null ;
 SvgMap.where  	 	= null ;
 SvgMap.oldPosX	 	= 0 ;
 SvgMap.oldPosY	 	= 0 ;
 SvgMap.panning	 	= 0 ;
 SvgMap.gotResponse	= 0 ;
 SvgMap.timeOutHandle ;

 //____________________________________________________________________________
 SvgMap.init = function()
 {
  SvgMap.theClipArea         = document.getElementById("clipArea") ;
  SvgMap.theViewText         = document.getElementById("currentViewText") ;
  SvgMap.theElementText      = document.getElementById("currentElementText") ;
  SvgMap.theSelectedText     = document.getElementById("selectedElementText") ;
  var theRefresh             = top.opener.document.getElementById("refreshInterval") ;
  var refreshInterval        = theRefresh.options[theRefresh.selectedIndex].value;
  SvgMap.theClipArea.addEventListener('DOMMouseScroll',  SvgMap.mouseScrollListener, false);
  SvgMap.theClipArea.addEventListener("mousedown",       SvgMap.mouseDownListener,   false);
  setTimeout( "SvgMap.updateTrackerMap()",1000) ; // Capture first data snapshot as soon as possibile
  setInterval("SvgMap.updateTrackerMap()",refreshInterval) ;

 }
 
 //____________________________________________________________________________
 SvgMap.updateTrackerMap = function()
 {
   var obj         = top.opener.document.getElementById("monitoring_element_list") ;
   var selME       = obj.options[obj.selectedIndex].value;
   obj             = top.opener.document.getElementById("TKMapContentType");
   var stype 	   = obj.options[obj.selectedIndex].value;
   var queryString = "RequestID=periodicTrackerMapUpdate";
   var url	   = WebLib.getApplicationURL2();
//   url    	  += "/Request?";
   url    	  += queryString;   
   url    	  += '&MEName='    + selME;
   url 		  += '&TKMapType=' + stype;
   var theColorMap = document.getElementById("theColorMap") ;
   var theAlarmMap = document.getElementById("theAlarmMap") ;
   if( stype == "Alarms" )
   {
    theColorMap.setAttribute("style","visibility: hidden;" ) ;
    theAlarmMap.setAttribute("style","visibility: visible;" ) ;
   } else {
    theColorMap.setAttribute("style","visibility: visible;" ) ;
    theAlarmMap.setAttribute("style","visibility: hidden;" ) ;
   }
//   var cCodeMEText  = document.getElementById("colorCodeME").textContent ;
   var cCodeMEField = document.getElementById("currentColorCodeME") ;
   cCodeMEField.setAttribute("value",stype+" for ME "+selME) ;

   WebLib.makeRequest(url, SvgMap.repaintTrackerMap);     
 }
 //____________________________________________________________________________
 SvgMap.changeRefreshInterval = function()
 {
  var theRefresh      = top.opener.document.getElementById("refreshInterval") ;
  var refreshInterval = theRefresh.options[theRefresh.selectedIndex].value;
//  DM_TraceWindow(SvgMap.thisFile,arguments.callee.name,"New refresh interval: "+refreshInterval) ;
 }
 //____________________________________________________________________________
 SvgMap.repaintTrackerMap = function()
 {
  if (WebLib.http_request.readyState == 4) 
  {
   if (WebLib.http_request.status == 200) 
   {
    try 
    {
     SvgMap.theElementText      = document.getElementById("currentElementText") ;
     var opaFlag    = document.getElementById('statisticsOpacity') ;
     var theMin     = document.getElementById('minEntries') ;
     var theMax     = document.getElementById('maxEntries') ;
     var doc        = WebLib.http_request.responseXML;
     var root       = doc.documentElement;
     var dets       = root.getElementsByTagName("DetInfo") ;
     var minEntries = 9999999999 ;
     var maxEntries = 0 ;
     var theMinId  ;
     var theMaxId  ;
     var theMinOpa ;
     var theMaxOpa ;
     var theMinID  ;
     var theMaxID  ;
     for (var i = 0; i < dets.length; i++) 
     {
      var detId      = dets[i].getAttribute("DetId") ;
      var red	     = dets[i].getAttribute("red"  ) ;
      var green      = dets[i].getAttribute("green") ;
      var blue       = dets[i].getAttribute("blue" ) ;
      var thePolygon = document.getElementById(detId) ;
      var rgb	     = "rgb(" + red + "," + green + "," + blue + ")" ;
      thePolygon.setAttribute("fill",rgb) ;
      var entries    = parseInt(dets[i].getAttribute("entries" )) ;
      var opacity    = parseFloat(entries) / 100 + .1 ;
      if( opacity > 1 || !opaFlag.checked) {opacity = 1;}
      thePolygon.setAttribute("style","fill-opacity: "+opacity) ;
      thePolygon.setAttribute("entries",entries) ;
      if( entries < minEntries ) 
      {
       minEntries = entries ;
       theMinId   = thePolygon ;
       theMinOpa  = opacity ;
       theMinID   = detId ;
      }
      if( entries > maxEntries ) 
      {
       maxEntries = entries ;
       theMaxId   = thePolygon ;
       theMaxOpa  = opacity ;
       theMaxID   = detId ;
      }
     }
     try
     {
      theMinId.setAttribute("style","fill-opacity: "+theMinOpa+"; stroke: black; stroke-width: 4") ;
      theMaxId.setAttribute("style","fill-opacity: "+theMaxOpa+"; stroke: red;	 stroke-width: 4") ;
     } catch(e) {}
     theMin.textContent = "Min: " + minEntries + " ("+theMinID+")";
     theMax.textContent = "Max: " + maxEntries + " ("+theMaxID+")";
     var normTag     = root.getElementsByTagName("theLimits") ;
     var normLow     = parseFloat(normTag[0].getAttribute("normLow" )) ;
     var normHigh    = parseFloat(normTag[0].getAttribute("normHigh")) ;
     var deltaNorm   = (normHigh - normLow)/5 ;
     var tagName     = "colorCodeMark" ;
     try 
     {
      for( var i=5; i>=0; i-- )
      {
       tagName = "colorCodeMark" + i ;
       var markTag = document.getElementById(tagName) ;
       markTag.textContent = parseInt(i * deltaNorm) ;
      }
     } catch(e) {}
     SvgMap.gotResponse = 1 ;
    } catch(error) {
     alert("[.SvgMap.js::SvgMap.repaintTrackerMap()] Error: " + error.message +
           "\n\nMost likely this means the web-server died or we lost connection with it") ;
    }
   }
  }
 }
 //_____________________________________________MEName_______________________________
 SvgMap.mouseScrollListener = function(evt)
 {
  if (evt.detail) 
  {
    SvgMap.zoomAmount = Math.abs(evt.detail / 3 * SvgMap.theZoomAmount) ;
    if( evt.detail > 0 )
    {
     SvgMap.zoomIt("In") ;
    } else {
     SvgMap.zoomIt("Out") ;
    }
  }
 }

 //____________________________________________________________________________
 SvgMap.mouseDownListener = function(evt)
 {
  SvgMap.panning = 1 ;
  SvgMap.oldPosX = evt.clientX ;
  SvgMap.oldPosY = evt.clientY ;
  SvgMap.theClipArea.setAttribute("style","cursor: move;");
  document.addEventListener("mousemove", SvgMap.mouseMoveListener, true);
  document.addEventListener("mouseup",   SvgMap.mouseUpListener,   true);
 }
 
 //____________________________________________________________________________
 SvgMap.mouseMoveListener = function(evt)
 {
  var stepTolerance = 1 ;
  if( SvgMap.panning == 1 )
  {
   var deltaX = evt.clientX - SvgMap.oldPosX ;
   var deltaY = evt.clientY - SvgMap.oldPosY ;
   SvgMap.oldPosX    = evt.clientX ;
   SvgMap.oldPosY    = evt.clientY ;
   if( deltaX > stepTolerance && Math.abs(deltaY) < stepTolerance)
   {
     SvgMap.zoomIt("Right") ;
     return ;
   } else if ( deltaX < -stepTolerance && Math.abs(deltaY) < stepTolerance){
     SvgMap.zoomIt("Left") ;
     return ;
   } 
   if( deltaY > stepTolerance && Math.abs(deltaX) < stepTolerance )
   {
     SvgMap.zoomIt("Down") ;
     return ;
   } else if( deltaY < -stepTolerance && Math.abs(deltaX) < stepTolerance ){
     SvgMap.zoomIt("Up") ;
     return ;
   } 
  } else {
  }
 }
 
 //____________________________________________________________________________
 SvgMap.mouseUpListener = function(evt)
 {
  SvgMap.panning = 0;
  SvgMap.theClipArea.setAttribute("style","cursor: default;");
 }
 
 //____________________________________________________________________________
 SvgMap.showData = function (evt)
 {
  var xlinkns = "http://www.w3.org/1999/xlink"; 
  var currentMEList = new Array() ;
  var currentMESrc  = new Array() ;
  SvgMap.where  = evt.currentTarget;

  if (evt.type == "click") //   <-------------------------------- C l i c k -------
  {
   SvgMap.drawMarker("black") ;
   var leftDoc  = top.left.document ;  
   var rightDoc = top.right.document ; // Fetch a pointer to the right frame
      
   var theImages                  = new Array() ;
   var theRightInnerFrame         = top.right.frames ;
   var theRightInnerFrameElements = theRightInnerFrame[0].document.getElementsByTagName("div") ;

   var myPoly   = evt.currentTarget;
   var moduleId = myPoly.getAttribute("detid"); 
   try
   {
    var destURL;
    var theMEList = top.opener.document.getElementById("monitoring_element_list") ;
    var selME     =  theMEList.options[theMEList.selectedIndex].value;
    var destId    = 0 ;
    for( var i=0; i < theMEList.length; i++)
    {
      var myTrackerPlot ;
      if( theMEList[i].value == selME ) 
      {      
       destURL = "baseImage0" ;
       myTrackerPlot = top.right.document.getElementById(destURL);
      } else {
       destURL = "baseImage" + ++destId;
       myTrackerPlot = theRightInnerFrame[0].document.getElementById(destURL);
      }
<<<<<<< svgmap.js
      var url_serv = WebLib.getApplicationURL2();
      //var url_serv      = "http://lxplus096.cern.ch:1972/urn:xdaq-application:lid=15/Request?";
=======
      var url_serv      = "http://lxplus202.cern.ch:1972/urn:xdaq-application:lid=15/Request?";
>>>>>>> 1.6
      var queryString   = "RequestID=PlotTkMapHistogram";
      queryString      += "&ModId="  + moduleId;
      queryString      += "&MEName=" + theMEList[i].value;
      var url1          = url_serv   + queryString;
      myTrackerPlot.setAttribute("src", url1);
//      SvgMap.smallDelay(3000);
//      queryString       = "RequestID=UpdateTkMapPlot" ;
//      queryString      += "&ModId="  + moduleId;
//      queryString      += "&MEName=" + theMEList[i].value;
//      var url2          = url_serv   + queryString;
//      myTrackerPlot.setAttribute("src", url2);  
      currentMEList[i] = "ME: " + theMEList[i].value + " Id: " + moduleId ;
      currentMESrc[i]  = myTrackerPlot.getAttribute("src") ;
    }
   } catch(error) {
    alert("[SvgMap.js::"+arguments.callee.name+"] Fatal: "+error.message) ;
   }

   SvgMap.theSelectedText.setAttribute("value",SvgMap.where.getAttribute("POS")) ;
   
   var innerPlots = currentMEList.length - 1 ; // Last one is the summary plot
   // Push the list of ME names into the combobox in the right frame
   var theMEListSelectors = top.right.document.forms['MEListForm'].MEListCB ;
   
   for( var i=0; i < currentMEList.length; i++)
   {
    theMEListSelectors.options[i] = new Option(currentMEList[i],currentMESrc[i]);
   }
  }

  if (evt.type == "mouseover") //   <----------------------------------------------- 
  {
    var theStyle = SvgMap.where.getAttribute("style") ;
    try
    {
     var opacity  = theStyle.match(/fill-opacity:\\s+(\\d+)/) ;
     theStyle     = "cursor:crosshair; fill-opacity: " + opacity ;
    } catch(error) {
     theStyle     = "cursor:crosshair; fill-opacity: 1" ;
    } 
    SvgMap.where.setAttribute("style",theStyle) ;
    SvgMap.theElementText.setAttribute("value",SvgMap.where.getAttribute("POS")+
                                               " -- Entries:" + 
					       SvgMap.where.getAttribute("entries")) ;
  }

  if (evt.type == "mouseout")  //   <-----------------------------------------------
  {
   SvgMap.theElementText.setAttribute("value","-") ;
  }
 }

 //------------------------------------------------------------------------------------------
 SvgMap.entriesOpacity = function()
 {
  var polygons = document.getElementsByTagName("polygon") ;
  for( var i=0; i<polygons.length; i++)
  {
    polygons[i].setAttribute("style","") ;
  }
 }
 //------------------------------------------------------------------------------------------
 SvgMap.smallDelay = function(millis)
 {
   var inizio = new Date();
   var inizioint=inizio.getTime();
   var intervallo = 0;
   while(intervallo<millis)
   {
     var fine = new Date();
     var fineint=fine.getTime();
     intervallo = fineint-inizioint;
   }
 }
 
 //____________________________________________________________________________
 SvgMap.drawMarker = function(color) 
 {
  document.getElementById("spot").setAttribute("points",SvgMap.where.getAttribute("points")) ;
  document.getElementById("spot").setAttribute("fill",color) ;
 }

 //____________________________________________________________________________
 SvgMap.changeContentType = function()
 {
 }
 
 //____________________________________________________________________________
 SvgMap.zoomIt = function(what)
 {
  var vBAtt = SvgMap.theClipArea.getAttribute("viewBox") ;
  var geo   = vBAtt.split(/\s+/) ;

  SvgMap.theViewText.setAttribute("value",what) ;
  switch (what) 
  {
   case "FPIX1-z":
       geo[0]=  -30 ;
       geo[1]= -250 ;
       geo[2]= 1200 ;
       geo[3]= 1200 ;
       break;
   case "FPIX2-z":
       geo[0]=  -30 ;
       geo[1]=  490 ;
       geo[2]= 1200 ;
       geo[3]= 1200 ;
       break;
   case "BPIX1":
       geo[0]=  475 ;
       geo[1]=    0 ;
       geo[2]= 1495 ;
       geo[3]=  795 ;
       break;
   case "BPIX2-3":
       geo[0]=  500 ;
       geo[1]=  870 ;
       geo[2]= 1369 ;
       geo[3]=  727 ;
       break;
   case "FPIX1+z":
       geo[0]= 1285 ;
       geo[1]= -250 ;
       geo[2]= 1200 ;
       geo[3]= 1200 ;
       break;
   case "FPIX2+z":
       geo[0]= 1285 ;
       geo[1]=  540 ;
       geo[2]= 1200 ;
       geo[3]= 1100 ;
       break;
   case "Home":
       geo[0]=    0 ;
       geo[1]=    0 ;
       geo[2]= 3000 ;
       geo[3]= 1600 ;
       break;
   case "In":
       geo[2]= parseFloat(geo[2]) / SvgMap.zoomAmount ;
       geo[3]= parseFloat(geo[3]) / SvgMap.zoomAmount ;
       break;
   case "Out":
       geo[2]= parseFloat(geo[2]) * SvgMap.zoomAmount ;
       geo[3]= parseFloat(geo[3]) * SvgMap.zoomAmount ;
       break;
   case "Up":
       geo[1]= parseInt(geo[1])   + SvgMap.stepAmount ;
       break;
   case "Down":
       geo[1]= parseInt(geo[1])   - SvgMap.stepAmount ;
       break;
   case "Left":
       geo[0]= parseInt(geo[0])   + SvgMap.stepAmount ;
       break;
   case "Right":
       geo[0]= parseInt(geo[0])   - SvgMap.stepAmount ;
       break;
  }
  var newGeo = geo[0]+" "+geo[1]+" "+parseInt(geo[2])+" "+parseInt(geo[3]);
  SvgMap.theClipArea.setAttribute("viewBox",newGeo) ;
  SvgMap.showIt() ;  
 }
 //____________________________________________________________________________
 SvgMap.hideIt = function(evt)
 {
  SvgMap.where = evt.currentTarget;
  var theStyle = SvgMap.where.getAttribute("style") ;
  if( theStyle.match(/hidden/)) 
  {
   return ;
  }
  theStyle    += " visibility: hidden;" ;
  SvgMap.where.setAttribute("style", theStyle) ;
 }
 //____________________________________________________________________________
 SvgMap.showIt = function()
 {
  var where = document.getElementsByTagName("text");
  for( var i=0; i<where.length; i++)
  {
   if( where[i].getAttribute("name") == "overlappingDetectorLabel" )
   {
    var theStyle = where[i].getAttribute("style") ;
    if( theStyle.match(/visible/)) 
    {
     return ;
    }
    theStyle    += " visibility: visible;" ;
    where[i].setAttribute("style", theStyle) ;
   }
  }
 }

