  var thisFile  = "rightEmbedded.js" ;
  var theLenses = new Array() ;

  initializeRightInnerFrame() ;			    

  //================================================================
  function initializeRightInnerFrame()
  {
   var baseVect = document.getElementsByTagName("div") ;
   for( var i=0; i < baseVect.length; i++)
   {
    var name = baseVect[i].getAttribute("name") ;
    if( name == null ) {continue;}
    var m = baseVect[i].getAttribute("name").match(/(binding\w+)/) ;
    if( m.length > 0)
    {
     var lens = new DLMLens(m[0]		  , 
			    "images/EmptyPlot.png", 
			    1980		  , 
			    1530		 );
     theLenses.push(lens) ;		    
    }
   }
  }
  //================================================================
  function innerLoading(where,what)
  {
//   var lensScale = document.getElementById("lensScaling") ;
//   var selIndex  = lensScale.selectedIndex;
//   var scale     = parseFloat(lensScale.options[selIndex].value) ;

   var scale     = 1 ;
   var thisImg   = document.getElementById(what) ;
   var imgSrc    = thisImg.getAttribute("src");
   var geometry  = thisImg.getAttribute("alt") ;
   var parts     ;
   var size      ;
   var width     = 737 ; // Provide suitable defaults in case this snippet get executed
   var height    = 563 ; // before the alt tag is available (onload could have been deferred) 
   if( geometry != null ) 
   {
    parts     = geometry.split(":") ;
    size      = parts[1].split("x") ;
    width     = size[0] * scale ;
    height    = size[1] * scale ;
   }
   for( var i=0; i < theLenses.length; i++)
   {
     theLenses[i].update(where  , 
     		    	 imgSrc , 
     		    	 width  , 
     		    	 height);
   }
  }
  
  //================================================================
  function innerTransport(event)
  {
   var rightDoc = top.right.document ; // Fetch a pointer to the right frame
   var thePlaceHolder = rightDoc.getElementById("placeHolder") ;
   thePlaceHolder.setAttribute("src",event.target.src) ;
//   DM_TraceWindow(thisFile,arguments.callee.name,"New image: "+event.target.src) ;  
  }
  
