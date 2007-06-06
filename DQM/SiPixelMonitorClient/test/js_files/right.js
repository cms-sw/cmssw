  var thisFile  = "right.js" ;
  var theLenses = new Array() ;

  initializeRightFrame() ;			    

  //================================================================
  function initializeRightFrame()
  {
   var baseVect = document.getElementsByTagName("div") ;
   for( var i=0; i < baseVect.length; i++)
   {
    var name = baseVect[i].getAttribute("name") ;
    if( name == null ) {continue;}
    var m = baseVect[i].getAttribute("name").match(/(binding\w+)/) ;
    if( m.length > 0)
    {
     var lens = new DLMLens(m[0]		, 
			  "images/EmptyPlot.png", 
			  1600  		, 
			  1200  	       );
     theLenses.push(lens) ;		  
    }
   }
  }
  //================================================================
  function changeLensScaling()
  {
   var images = document.getElementsByTagName("img") ;
   for (var i=0; i < images.length; i++)
   {
    var parent=images[i].parentNode.id ;
    loading(parent,images[i].name) ;
   }
  }
			    
  //================================================================
  function loading(where,what)
  {
//   var lensScale = document.getElementById("lensScaling") ;
//   var selIndex  = lensScale.selectedIndex;
//   var scale     = parseFloat(lensScale.options[selIndex].value) ;

   var scalex    = 1 ;
   var scaley    = 1 ;
   var thisImg   = document.getElementById(what) ;
   var imgSrc    = thisImg.getAttribute("src");
   var geometry  = thisImg.getAttribute("alt") ;
   var parts     ;
   var size      ;
   var width     = 1600 ; // Provide suitable defaults in case this snippet gets executed
   var height    = 1200 ; // before the alt tag is available (onload could have been deferred) 
   if( geometry != null && geometry != "" ) 
   {
    parts  = geometry.split(":") ;
    size   = parts[1].split("x") ;
    width  = size[0] * scalex ;
    height = size[1] * scaley ;
   }
   for( var i=0; i < theLenses.length; i++)
   {
     theLenses[i].update(where  , 
		         imgSrc , 
		         width  , 
		         height);
   }

//   var onMo       = thisImg.getAttribute("onmouseover") ;
//   var pieces     = onMo.split(/return escape/) ;
//   var toolTipAtt = pieces[0] ;
//   var imgNoDir   = imgSrc.split(/images\//) ;
//   var imgName    = imgNoDir[1];
//var imgName = "<img src=\\'http://hal9000.mib.infn.it/~menasce/myTests/TrackerMapMockup/" + imgSrc + "\\' width=\\'300\\'>" ;
//   var newToolTip0 = toolTipAtt + "return escape('<img src=\\'http://hal9000.mib.infn.it/~menasce/myTests/TrackerMapMockup/images/" + imgName + "\\' width=\\'300\\'><br>Reference plot for this ME');" ;
//   var newToolTip = toolTipAtt + "return escape('<img src=\\'http://hal9000.mib.infn.it/~menasce/myTests/TrackerMapMockup/images/SCurve_Chip_0_Row_0_Col_0.png\\' width=\\'300\\'><br>Reference plot for this ME');" ;
// DM_TraceWindow(thisFile,arguments.callee.name,"l0:l "+newToolTip0.length+":"+newToolTip.length) ;  
//for( var i=0; i < newToolTip.length; i++)
//{
// if(newToolTip0.charAt(i) != newToolTip.charAt(i) )
// {
//  DM_TraceWindow(thisFile,arguments.callee.name,i+"] "+newToolTip0.charAt(i)+" <-> " +newToolTip.charAt(i)) ;  
// }
//}
//   DM_TraceWindow(thisFile,arguments.callee.name,newToolTip0) ;  
//   DM_TraceWindow(thisFile,arguments.callee.name,newToolTip) ;  
//   thisImg.setAttribute("onmouseover",newToolTip0) ;
  }
  
  //================================================================
  function transport(event)
  {
   dd.elements.zoomedImg.swapImage(event.target.src) ;
//   DM_TraceWindow("right.js",arguments.callee.name,"New image: "+dd.elements.zoomedImg.getAttribute("src")) ;  
  }
  
  //================================================================
  function resizeToDefault()
  {
   var zoomedImg = document.getElementById("zoomedImg") ;
   var width     = zoomedImg.getAttribute("width") ;
   var height    = zoomedImg.getAttribute("height") ;
   dd.elements.zoomedImg.resizeTo(width,height) ;
   dd.elements.zoomedImg.moveTo(findPosX(zoomedImg),findPosY(zoomedImg)) ;
  }
  //================================================================
  function resizeByStep()
  {
   var currW = dd.elements.zoomedImg.w ;
   var currH = dd.elements.zoomedImg.h ;
   var theResizeSelection = top.right.document.forms['resizeByForm'].resizeBy ;
   var theSelIndex        = theResizeSelection.selectedIndex ;
   var theSelValue        = theResizeSelection[theSelIndex].value ;
   dd.elements.zoomedImg.resizeTo(currW * theSelValue, currH * theSelValue) ;
  }
  //================================================================
  function swapImage()
  {
//    var thePlaceHolder = document.getElementById("placeHolder") ;
//    var newImg = thePlaceHolder.getAttribute("src") ;
//    dd.elements.zoomedImg.swapImage(newImg) ;
//    DM_TraceWindow("right.js",arguments.callee.name,"New image: "+newImg) ;  
  }
  //================================================================
  function movePlotTo()
  {
   var theRightInnerFrame       = top.right.frames ;
   var theRightInnerFrameWindow = theRightInnerFrame[0] ;
   var theMEListSelections      = top.right.document.forms['MEListForm'].MEListCB ;
   var theSelIndex              = theMEListSelections.selectedIndex ;
   var theSelName               = theMEListSelections[theSelIndex].value ;
   var baseVect                 = theRightInnerFrameWindow.document.getElementsByTagName("div") ;
   var posX ;
   var posY ;
   for( var i=0; i < baseVect.length; i++)
   {
    var name = baseVect[i].getAttribute("name") ;
    if( name == null ) {continue;}
    var m = name.match(/binding(\w+)/) ;
    if( m.length > 0 )
    {
     var imgRefs = baseVect[i].childNodes ;
     for( var j=0; j < imgRefs.length; j++)
     {
      if( imgRefs[j].tagName == "IMG" )
      {
       var imgSrc = imgRefs[j].getAttribute("src") ;
//       if( imgSrc == "images/"+theSelName )
       if( imgSrc == theSelName )
       {
        posX = findPosX(imgRefs[j]) ;
        posY = findPosY(imgRefs[j]) ;
       }
      }
     }
    }
   }
   theRightInnerFrameWindow.scrollTo(posX,posY) ;
  }
  //================================================================
  SET_DHTML(CURSOR_MOVE, RESIZABLE, TRANSPARENT, "zoomedImg");
