var RightEmbedded = {} ;

RightEmbedded.thisFile  = "rightEmbedded.js" ;
RightEmbedded.theLenses = new Array() ;
RightEmbedded.theLensNM = new Array() ;

//================================================================
RightEmbedded.initializeRightInnerFrame = function()
{
 var baseVect = document.getElementsByTagName("div") ;
 for( var i=0; i < baseVect.length; i++)
 {
  var name = baseVect[i].getAttribute("name") ;
  if( name == null ) {continue;}
  var m = baseVect[i].getAttribute("name").match(/(binding\w+)/) ;
  if( m.length > 0)
  {
   var lens = new DLMLens(m[0]  		, 
        		  "images/EmptyPlot.png", 
        		  1980  		, 
        		  1530  	       );
   RightEmbedded.theLenses.push(lens) ; 		  
   RightEmbedded.theLensNM.push(m[0]) ; 		  
  }
 }
}
//================================================================
RightEmbedded.innerLoading = function(where,what)
{
  var scale	= 1 ;
  var thisImg	= document.getElementById(what) ;
  var imgSrc	= thisImg.getAttribute("src");
  var geometry  = thisImg.getAttribute("alt") ;
  var parts	;
  var size	;
  var width	= 1600 ; // Provide suitable defaults in case this snippet get executed
  var height	= 1200 ; // before the alt tag is available (onload could have been deferred) 
//  if( geometry != null ) 
//  {
//   DM_TraceWindow("rightEmbedded.js",arguments.callee.name,"geometry "+geometry) ;  
//   parts     = geometry.split(":") ;
//   DM_TraceWindow("rightEmbedded.js",arguments.callee.name,"parts    "+parts) ;  
//   size      = parts[1].split("x") ;
//   width     = size[0] * scale ;
//   height    = size[1] * scale ;
//  }
 for( var i=0; i < RightEmbedded.theLenses.length; i++)
 {
   RightEmbedded.theLenses[i].update(where  , 
				     imgSrc , 
				     width  ,
				     height);		       
 }
}

//================================================================
RightEmbedded.innerTransport = function(event)
{
 top.right.dd.elements.zoomedImg.swapImage(event.target.src) ;
// DM_TraceWindow("rightEmbedded.js",arguments.callee.name,"Swapping to: "+event.target.src) ;  
}
  
//================== E x e c u t e ================================
RightEmbedded.initializeRightInnerFrame() ;			    

