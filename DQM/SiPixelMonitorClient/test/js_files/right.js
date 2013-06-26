var RightFrame = {} ;

RightFrame.thisFile  = "right.js" ;
RightFrame.theLenses = new Array() ;

//================================================================
RightFrame.initializeRightFrame = function()
{
 var baseVect = document.getElementsByTagName("div") ;

 for( var i=0; i < baseVect.length; i++)
 {
  var name = baseVect[i].getAttribute("name") ;
  if( name == null ) {continue;}
  var m = baseVect[i].getAttribute("name").match(/(binding\w+)/) ;
  if( m.length > 0)
  {
   var lens = new DLMLens(m[0]  	      , 
        		"images/EmptyPlot.png", 
        		1600		      , 
        		1200		     );
   RightFrame.theLenses.push(lens) ;		
  }
 }
}
//================================================================
RightFrame.changeLensScaling = function()
{
 var images = document.getElementsByTagName("img") ;
 for (var i=0; i < images.length; i++)
 {
  var parent=images[i].parentNode.id ;
  RightFrame.loading(parent,images[i].name) ;
 }
}
        		  
//================================================================
RightFrame.loading = function(where,what)
{
 try
 {
  var scalex    = 1 ;
  var scaley	= 1 ;
  var thisImg	= document.getElementById(what) ;
  var imgSrc	= thisImg.getAttribute("src");
  var geometry  = thisImg.getAttribute("alt") ;
  var parts	;
  var size	;
  var width	= 1600 ; // Provide suitable defaults in case this snippet gets executed
  var height	= 1200 ; // before the alt tag is available (onload could have been deferred) 
  if( geometry != null && geometry != "" ) 
  {
   parts  = geometry.split(":") ;
   size   = parts[1].split("x") ;
   width  = size[0] * scalex ;
   height = size[1] * scaley ;
  }
  for( var i=0; i < RightFrame.theLenses.length; i++)
  {
    RightFrame.theLenses[i].update(where  , 
  				   imgSrc ,
  				   width  ,
  				   height);
  }
 } catch(error) {
  alert("[right.js::RightFrame.loading()] Fatal syntax/execution error: "+error.message) ;
 }
//   var onMo       = thisImg.getAttribute("onmouseover") ;
//   var pieces     = onMo.split(/return escape/) ;
//   var toolTipAtt = pieces[0] ;
//   var imgNoDir   = imgSrc.split(/images\//) ;
//   var imgName    = imgNoDir[1];
//var imgName = "<img src=\\'http://hal9000.mib.infn.it/~menasce/myTests/TrackerMapMockup/" + imgSrc + "\\' width=\\'300\\'>" ;
//   var newToolTip0 = toolTipAtt + "return escape('<img src=\\'http://hal9000.mib.infn.it/~menasce/myTests/TrackerMapMockup/images/" + imgName + "\\' width=\\'300\\'><br>Reference plot for this ME');" ;
//   var newToolTip = toolTipAtt + "return escape('<img src=\\'http://hal9000.mib.infn.it/~menasce/myTests/TrackerMapMockup/images/SCurve_Chip_0_Row_0_Col_0.png\\' width=\\'300\\'><br>Reference plot for this ME');" ;
// DM_TraceWindow(RightFrame.thisFile,arguments.callee.name,"l0:l "+newToolTip0.length+":"+newToolTip.length) ;  
//for( var i=0; i < newToolTip.length; i++)
//{
// if(newToolTip0.charAt(i) != newToolTip.charAt(i) )
// {
//  DM_TraceWindow(RightFrame.thisFile,arguments.callee.name,i+"] "+newToolTip0.charAt(i)+" <-> " +newToolTip.charAt(i)) ;  
// }
//}
//   DM_TraceWindow(RightFrame.thisFile,arguments.callee.name,newToolTip0) ;  
//   DM_TraceWindow(RightFrame.thisFile,arguments.callee.name,newToolTip) ;  
//   thisImg.setAttribute("onmouseover",newToolTip0) ;
}

//================================================================
RightFrame.transport = function(event)
{
 dd.elements.zoomedImg.swapImage(event.target.src) ;
}

//================================================================
RightFrame.resizeToDefault = function()
{
 var zoomedImg = document.getElementById("zoomedImg") ;
 var width     = zoomedImg.getAttribute("width") ;
 var height    = zoomedImg.getAttribute("height") ;
 dd.elements.zoomedImg.resizeTo(width,height) ;
 dd.elements.zoomedImg.moveTo(findPosX(zoomedImg),findPosY(zoomedImg)) ;
}
//================================================================
RightFrame.resizeByStep = function()
{
 var currW = dd.elements.zoomedImg.w ;
 var currH = dd.elements.zoomedImg.h ;
 currW = 400 ;
 currH = 300 ;

 var theResizeSelection = top.right.document.forms['resizeByForm'].resizeBy ;
 var theSelIndex	= theResizeSelection.selectedIndex ;
 var theSelValue	= theResizeSelection[theSelIndex].value ;
 dd.elements.zoomedImg.resizeTo(currW * theSelValue, currH * theSelValue) ;
}
//================================================================
RightFrame.swapImage = function()
{
//    var thePlaceHolder = document.getElementById("placeHolder") ;
//    var newImg = thePlaceHolder.getAttribute("src") ;
//    dd.elements.zoomedImg.swapImage(newImg) ;
//    DM_TraceWindow("right.js",arguments.callee.name,"New image: "+newImg) ;  
}
//================================================================
RightFrame.movePlotTo = function()
{
 var theRightInnerFrame       = top.right.frames ;
 var theRightInnerFrameWindow = theRightInnerFrame[0] ;
 var theMEListSelections      = top.right.document.forms['MEListForm'].MEListCB ;
 var theSelIndex	      = theMEListSelections.selectedIndex ;
 var theSelName 	      = theMEListSelections[theSelIndex].value ;
 var baseVect		      = theRightInnerFrameWindow.document.getElementsByTagName("div") ;
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
RightFrame.externalImage = function()
{
 var zoomedImgSrc = document.getElementById("zoomedImg").getAttribute("src") ;
 var url	  = window.location.href.split("temporary") ;
 url              = url[0] + "temporary/Popper.html?" + zoomedImgSrc ;

 alert("[right.js::RightFrame.externalImage()] url: " +url ) ;

 var win = window.open(url,
 		       "popupWindowDario"  ,
 		       "menubar   = no,  " +
 		       "location  = no,  " +
 		       "resizable = no,  " +
 		       "scrollbars= yes, " +
 		       "titlebar  = no,  " +
 		       "status    = no,  " +
 		       "left	  =   0, " +
 		       "top	  =   0, " +
 		       "height    = 600, " +
 		       "width	  = 800 ") ;
 win.moveTo(100,100) ;
 win.focus();		 

}

//================================================================

// RightFrame.initializeRightFrame() ;			  

// SET_DHTML(CURSOR_MOVE, RESIZABLE, TRANSPARENT, "zoomedImg");


