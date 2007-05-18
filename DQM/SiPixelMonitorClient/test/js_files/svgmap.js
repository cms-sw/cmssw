 var thisFile        = "svgmap.js" ;
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
 var timeOutHandle ;

 var plots = new Array("images/Bump_Bonds_A_Test_Chip_0_Distribution_2D.png",
 		       "images/Bump_Bonds_A_Test_Chip_1_Distribution_2D.png",
 		       "images/Bump_Bonds_A_Test_Chip_2_Distribution_2D.png",
 		       "images/Bump_Bonds_A_Test_Chip_3_Distribution_2D.png",
 		       "images/Bump_Bonds_A_Test_Chip_4_Distribution_2D.png",
 		       "images/Bump_Bonds_A_Test_Chip_5_Distribution_2D.png",
 		       "images/Bump_Bonds_A_Test_Chip_6_Distribution_2D.png",
 		       "images/Bump_Bonds_A_Test_Chip_7_Distribution_2D.png",
 		       "images/Bump_Bonds_A_Test_Chip_8_Distribution_2D.png",
 		       "images/Bump_Bonds_A_Test_Chip_9_Distribution_2D.png",
        	       "images/picture.jpg",
 		       "images/SCurve_Chip_0_Row_0_Col_0.png",
 		       "images/SCurve_Chip_1_Row_1_Col_1.png",
 		       "images/SCurve_Chip_2_Row_2_Col_2.png",
 		       "images/SCurve_Chip_3_Row_3_Col_3.png",
 		       "images/SCurve_Chip_4_Row_4_Col_4.png",
 		       "images/SCurve_Chip_5_Row_5_Col_5.png",
 		       "images/SCurve_Chip_6_Row_6_Col_6.png",
 		       "images/SCurve_Chip_7_Row_7_Col_7.png",
 		       "images/SCurve_Chip_8_Row_8_Col_8.png",
        	       "images/SCurve_Chip_9_Row_9_Col_9.png"
 		      ) ;
 var plotsGeometry = new Array("1596x1172", "1596x1172", "1596x1172", "1596x1172", "1596x1172", 
 			       "1596x1172", "1596x1172", "1596x1172", "1596x1172", "1596x1172",
        		       "3648x2565",
        		       "1596x1172", "1596x1172", "1596x1172", "1596x1172", "1596x1172", 
 			       "1596x1172", "1596x1172", "1596x1172", "1596x1172", "1596x1172" 
 			      ) ;
 
  //____________________________________________________________________________
 function init()
 {
  DM_TraceWindow(thisFile,arguments.callee.name,"Initializer()") ;
  theClipArea     = document.getElementById("clipArea") ;
  theViewText     = document.getElementById("currentViewText") ;
  theElementText  = document.getElementById("currentElementText") ;
  theSelectedText = document.getElementById("selectedElementText") ;
  theClipArea.addEventListener('DOMMouseScroll',  mousescroll_listener, false);
  theClipArea.addEventListener("mousedown",       mousedown_listener,   false);
 }
 
  //____________________________________________________________________________
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
  where  = evt.currentTarget;

  if (evt.type == "click") //   <-----------------------------------------------
  {
   drawMarker("black") ;
   var leftDoc  = top.left.document ;  
   var rightDoc = top.right.document ; // Fetch a pointer to the right frame
      
   var theImages                  = new Array() ;
   var theRightInnerFrame         = top.right.frames ;
   var theRightInnerFrameElements = theRightInnerFrame[0].document.getElementsByTagName("div") ;

   var myPoly        = evt.currentTarget;
   var moduleId      = myPoly.getAttribute("detid"); 
   try
   {
    var theMEList = top.opener.document.getElementById("monitoring_element_list") ;
    for( var i=0; i < theMEList.length; i++)
    {
      var myTrackerPlot ;
      if( i == 0 ) 
      {
       myTrackerPlot = top.right.document.getElementById("baseImage0");
      } else {
       myTrackerPlot = theRightInnerFrame[0].document.getElementById("baseImage" + i);
      }
      var url_serv      = "http://lxplus211.cern.ch:1972/urn:xdaq-application:lid=15/Request?";
      var queryString   = "RequestID=PlotTkMapHistogram";
      queryString      += "&ModId="  + moduleId;
      queryString      += "&MEName=" + theMEList[i].value;
      var url1          = url_serv   + queryString;
      myTrackerPlot.setAttribute("src", url1);
      pausecomp(1000);
      DM_TraceWindow("svgmap.js",arguments.callee.name,url1) ;
      queryString       = "RequestID=UpdatePlot&t="+moduleId;
      queryString      += "&MEName=" + theMEList[i].value;
      var url2 = url_serv  + queryString;
      myTrackerPlot.setAttribute("src", url2);  
      DM_TraceWindow("svgmap.js",arguments.callee.name,"posting into "+"baseImage" + i) ;
      DM_TraceWindow("svgmap.js",arguments.callee.name,url2) ;
      currentMEList[0] = "ME: " + theMEList[i].value + " Id: " + moduleId ;
    }
   } catch(error) {
    alert("[svgmap.js::"+arguments.callee.name+"] Fatal: "+error.message) ;
   }

   theSelectedText.setAttribute("value",where.getAttribute("POS")) ;
   
   var innerPlots = currentMEList.length - 1 ; // Last one is the summary plot

   // Push the list of ME names into the combobox in the right frame
   var theMEListSelectors = top.right.document.forms['MEListForm'].MEListCB ;
   for( var i=0; i < currentMEList.length - 1; i++)
   {
    theMEListSelectors.options[i] = new Option(currentMEList[i],currentMEList[i]);
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
  DM_TraceWindow("svgmap.js",arguments.callee.name,"Honoring black dot drawing request...") ;
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
  DM_TraceWindow(thisFile,arguments.callee.name,newGeo) ;
  theClipArea.setAttribute("viewBox",newGeo) ;  
 }
