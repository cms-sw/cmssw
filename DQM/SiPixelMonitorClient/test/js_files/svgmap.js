 var thisFile        = ".svgmap.js" ;
 var theZoomAmount   = 1.05 ;
 var theStepAmount   = 25 ;
 var zoomAmount      = theZoomAmount ;
 var stepAmount      = theStepAmount ;
 var theViewText     = null ;
 var theElementText  = null ;
 var theSelectedText = null ;
 var theClipArea     = null ;
 var where           = null ;
 var oldPosX         = 0 ;
 var oldPosY         = 0 ;
 var panning         = 0 ;
 var gotResponse     = 0 ;
 var timeOutHandle ;

 //____________________________________________________________________________
 function init()
 {
  theClipArea         = document.getElementById("clipArea") ;
  theViewText         = document.getElementById("currentViewText") ;
  theElementText      = document.getElementById("currentElementText") ;
  theSelectedText     = document.getElementById("selectedElementText") ;
  var theRefresh      = top.opener.document.getElementById("refreshInterval") ;
  var refreshInterval = theRefresh.options[theRefresh.selectedIndex].value;
  theClipArea.addEventListener('DOMMouseScroll',  mousescroll_listener, false);
  theClipArea.addEventListener("mousedown",       mousedown_listener,   false);
  setInterval("updateTrackerMap()",refreshInterval) ;
//  DM_TraceWindow(thisFile,arguments.callee.name,"Initialized with refresh interval: "+refreshInterval) ;
 }
 
 //____________________________________________________________________________
 function updateTrackerMap()
 {
   DM_TraceWindow(thisFile,arguments.callee.name,"http_request.readyState="+http_request.readyState) ;
   if( http_request.readyState == 2 ) 
   {
    DM_TraceWindow(thisFile,arguments.callee.name,"Still waiting for an answer...") ;
    return ; // If previous submission got no answer, skip retry
   }
//   DM_TraceWindow(thisFile,arguments.callee.name,"Udating...") ;
   var theMEList   = top.opener.document.getElementById("monitoring_element_list") ;
   var selME       =  theMEList.options[theMEList.selectedIndex].value;
   var queryString = "RequestID=periodicTrackerMapUpdate";
   var url = getApplicationURL2();
   url    += "/Request?";
   url    += queryString;   
   url    += '&MEName='+selME;
   makeRequest(url, repaintTrackerMap);     
 }
 //____________________________________________________________________________
 function changeRefreshInterval()
 {
  var theRefresh      = top.opener.document.getElementById("refreshInterval") ;
  var refreshInterval = theRefresh.options[theRefresh.selectedIndex].value;
//  DM_TraceWindow(thisFile,arguments.callee.name,"New refresh interval: "+refreshInterval) ;
 }
 //____________________________________________________________________________
 function repaintTrackerMap()
 {
  if (http_request.readyState == 4) 
  {
   if (http_request.status == 200) 
   {
    try 
    {
     var doc  = http_request.responseXML;
     var root = doc.documentElement;
     var dets = root.getElementsByTagName("DetInfo") ;
     for (var i = 0; i < dets.length; i++) 
     {
      var detId      = dets[i].getAttribute("DetId") ;
      var red	     = dets[i].getAttribute("red"  ) ;
      var green      = dets[i].getAttribute("green") ;
      var blue       = dets[i].getAttribute("blue" ) ;
      var thePolygon = document.getElementById(detId) ;
      var rgb	     = "rgb(" + red + "," + green + "," + blue + ")" ;
      thePolygon.setAttribute("fill",rgb) ;
     }
     var normTag     = root.getElementsByTagName("theLimits") ;
     var normLow     = parseFloat(normTag[0].getAttribute("normLow" )) ;
     var normHigh    = parseFloat(normTag[0].getAttribute("normHigh")) ;
     var deltaNorm   = (normHigh - normLow)/5 ;
     var tagName     = "colorCodeMark" ;
     for( var i=5; i>=0; i-- )
     {
      tagName = "colorCodeMark" + i ;
      var markTag = document.getElementById(tagName) ;
      markTag.textContent = parseInt(i * deltaNorm) ;
     }
     gotResponse = 1 ;
    } catch(error) {
     alert("[.svgmap.js::repaintTrackerMap()] Error: " + error.message) ;
    }
   }
  }
 }
 //_____________________________________________MEName_______________________________
 function mousescroll_listener(evt)
 {
  if (evt.detail) 
  {
    zoomAmount = Math.abs(evt.detail / 3 * theZoomAmount) ;
    if( evt.detail > 0 )
    {
     zoomIt("In") ;
    } else {
     zoomIt("Out") ;
    }
  }
 }

 //____________________________________________________________________________
 function mousedown_listener(evt)
 {
  panning = 1 ;
  oldPosX = evt.clientX ;
  oldPosY = evt.clientY ;
  theClipArea.setAttribute("style","cursor: move;");
  document.addEventListener("mousemove", mousemove_listener, true);
  document.addEventListener("mouseup",   mouseup_listener,   true);
 }
 
 //____________________________________________________________________________
 function mousemove_listener(evt)
 {
  var stepTolerance = 1 ;
  if( panning == 1 )
  {
   var deltaX = evt.clientX - oldPosX ;
   var deltaY = evt.clientY - oldPosY ;
   oldPosX    = evt.clientX ;
   oldPosY    = evt.clientY ;
   if( deltaX > stepTolerance && Math.abs(deltaY) < stepTolerance)
   {
     zoomIt("Right") ;
     return ;
   } else if ( deltaX < -stepTolerance && Math.abs(deltaY) < stepTolerance){
     zoomIt("Left") ;
     return ;
   } 
   if( deltaY > stepTolerance && Math.abs(deltaX) < stepTolerance )
   {
     zoomIt("Down") ;
     return ;
   } else if( deltaY < -stepTolerance && Math.abs(deltaX) < stepTolerance ){
     zoomIt("Up") ;
     return ;
   } 
  } else {
  }
 }
 
 //____________________________________________________________________________
 function mouseup_listener(evt)
 {
  panning = 0;
  theClipArea.setAttribute("style","cursor: default;");
 }
 
 //____________________________________________________________________________
 function showData(evt)
 {
  var xlinkns = "http://www.w3.org/1999/xlink"; 
  var currentMEList = new Array() ;
  var currentMESrc  = new Array() ;
  where  = evt.currentTarget;

  if (evt.type == "click") //   <-----------------------------------------------
  {
   drawMarker("black") ;
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
      var url_serv      = "http://lxplus213.cern.ch:1972/urn:xdaq-application:lid=15/Request?";
      var queryString   = "RequestID=PlotTkMapHistogram";
      queryString      += "&ModId="  + moduleId;
      queryString      += "&MEName=" + theMEList[i].value;
      var url1          = url_serv   + queryString;
      myTrackerPlot.setAttribute("src", url1);
      pausecomp(1000);
      queryString       = "RequestID=UpdatePlot" ;
      queryString      += "&ModId="  + moduleId;
      queryString      += "&MEName=" + theMEList[i].value;
      var url2          = url_serv   + queryString;
      myTrackerPlot.setAttribute("src", url2);  
      currentMEList[i] = "ME: " + theMEList[i].value + " Id: " + moduleId ;
      currentMESrc[i]  = myTrackerPlot.getAttribute("src") ;
    }
   } catch(error) {
    alert("[svgmap.js::"+arguments.callee.name+"] Fatal: "+error.message) ;
   }

   theSelectedText.setAttribute("value",where.getAttribute("POS")) ;
   
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
    where.setAttribute("style","cursor:crosshair;") ;
    theElementText.setAttribute("value",where.getAttribute("POS")) ;
  }

  if (evt.type == "mouseout")  //   <-----------------------------------------------
  {
   theElementText.setAttribute("value","-") ;
  }
 }

 //------------------------------------------------------------------------------------------
 function pausecomp(millis)
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
 function drawMarker(color) 
 {
  document.getElementById("spot").setAttribute("points",where.getAttribute("points")) ;
  document.getElementById("spot").setAttribute("fill",color) ;
 }

 //____________________________________________________________________________
 function zoomIt(what)
 {
  var vBAtt = theClipArea.getAttribute("viewBox") ;
  var geo   = vBAtt.split(/\s+/) ;

  theViewText.setAttribute("value",what) ;
  switch (what) 
  {
   case "FPIX1-z":
       geo[0]=   30 ;
       geo[1]=   50 ;
       geo[2]=  593 ;
       geo[3]=  307 ;
       break;
   case "FPIX2-z":
       geo[0]=   30 ;
       geo[1]=  450 ;
       geo[2]=  593 ;
       geo[3]=  307 ;
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
       geo[0]= 1525 ;
       geo[1]=   45 ;
       geo[2]=  686 ;
       geo[3]=  309 ;
       break;
   case "FPIX2+z":
       geo[0]= 1525 ;
       geo[1]=  445 ;
       geo[2]=  686 ;
       geo[3]=  309 ;
       break;
   case "Home":
       geo[0]=    0 ;
       geo[1]=    0 ;
       geo[2]= 3000 ;
       geo[3]= 1600 ;
       break;
   case "In":
       geo[2]= parseFloat(geo[2]) / zoomAmount ;
       geo[3]= parseFloat(geo[3]) / zoomAmount ;
       break;
   case "Out":
       geo[2]= parseFloat(geo[2]) * zoomAmount ;
       geo[3]= parseFloat(geo[3]) * zoomAmount ;
       break;
   case "Up":
       geo[1]= parseInt(geo[1])   + stepAmount ;
       break;
   case "Down":
       geo[1]= parseInt(geo[1])   - stepAmount ;
       break;
   case "Left":
       geo[0]= parseInt(geo[0])   + stepAmount ;
       break;
   case "Right":
       geo[0]= parseInt(geo[0])   - stepAmount ;
       break;
  }
  var newGeo = geo[0]+" "+geo[1]+" "+parseInt(geo[2])+" "+parseInt(geo[3]);
  theClipArea.setAttribute("viewBox",newGeo) ;  
 }
