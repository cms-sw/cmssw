//_____________________________________________________________________
// Author: D. Menasce                                                  |
//                                                                     |
//_____________________________________________________________________|

var IMGC = {} ;

IMGC.IMAGE_LIST_URL      = '/temporary/images/filesList.lis';
IMGC.IMAGE_LIST_TITLES   = '/temporary/images/filesTitles.lis';
IMGC.PATH_TO_PICTURES    = '/temporary/images/';
IMGC.IMAGES_PER_ROW      = 2;
IMGC.IMAGES_PER_COL      = 2;
IMGC.IMAGES_PER_PAGE     = IMGC.IMAGES_PER_ROW * IMGC.IMAGES_PER_COL;
IMGC.THUMB_MICROFICATION = 4;
IMGC.INACTIVE_OPACITY    = 0.5;
IMGC.DEF_IMAGE_WIDTH     = 600;
IMGC.ASPECT_RATIO        = 1.5 ;
IMGC.BASE_IMAGE_WIDTH    = IMGC.DEF_IMAGE_WIDTH;
IMGC.BASE_IMAGE_HEIGHT   = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ;
IMGC.THUMB_IMAGE_WIDTH   = IMGC.BASE_IMAGE_WIDTH  / IMGC.IMAGES_PER_ROW;
IMGC.THUMB_IMAGE_HEIGHT  = IMGC.BASE_IMAGE_HEIGHT / IMGC.IMAGES_PER_COL;
IMGC.GLOBAL_RATIO        = .6 ;
IMGC.lastSource          = "" ;

//__________________________________________________________________________________________________________________________________
// Subscribe the current window for an onLoad signal.
// This will trigger an Ajax request to the server to get the list of available
// plots, and will further subscribe buttons to their appropriate call-back function
//
Event.observe(window, 'load', function()	
{
 IMGC.loadingProgress('visible') ;
 
 IMGC.getImageList() ;
 
// Event.observe($('loadBatch'),    'click', IMGC.loadBatch,    false);
// Event.observe($('reloadBatch'),  'click', IMGC.getImageList, false);

// Event.observe($('headerText'),   'mouseover', function()	
// {
//  IMGC.changeHeaderOpacity('mouseover');
// }, false);
// Event.observe($('headerText'),   'mouseout',  function()	
// {
//  IMGC.changeHeaderOpacity('mouseout');
// }, false);
 Event.observe($('firstPage'),     'click',     function()
 {
  IMGC.updatePage('first');
 }, false);
 Event.observe($('previousPage'),  'click',     function()
 {
  IMGC.updatePage('previous');
 }, false);
 Event.observe($('nextPage'),      'click',     function()
 {
  IMGC.updatePage('next');
 }, false);
 Event.observe($('lastPage'),      'click',     function()
 {
  IMGC.updatePage('last');
 }, false);
 Event.observe($('normalSize'),    'click',     function()
 {
  IMGC.changeSize('=');
 }, false);
 Event.observe($('smallerSize'),   'click',     function()
 {
  IMGC.changeSize('-');
 }, false);
 Event.observe($('largerSize'),    'click',     function()
 {
  IMGC.changeSize('+');
 }, false);
 Event.observe($('maximizeSize'),  'click',     function()
 {
  IMGC.changeSize('M');
 }, false);
 Event.observe($('oneXone'),       'click',     function()
 {
  IMGC.IMAGES_PER_ROW      = 1 ;
  IMGC.IMAGES_PER_COL      = 1;
  IMGC.IMAGES_PER_PAGE     = IMGC.IMAGES_PER_ROW * IMGC.IMAGES_PER_COL;
  IMGC.IMAGES_PER_PAGE     = 1 ;
  IMGC.ASPECT_RATIO        = 1.5 ;
  IMGC.THUMB_MICROFICATION = 4 ;
  IMGC.BASE_IMAGE_WIDTH    = IMGC.DEF_IMAGE_WIDTH ;
  IMGC.BASE_IMAGE_HEIGHT   = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO);
  IMGC.setBorderSize() ;
  IMGC.paintImages();
 }, false);
 Event.observe($('twoXtwo'),       'click',     function()
 {
  IMGC.IMAGES_PER_ROW      = 2 ;
  IMGC.IMAGES_PER_COL      = 2;
  IMGC.IMAGES_PER_PAGE     = IMGC.IMAGES_PER_ROW * IMGC.IMAGES_PER_COL;
  IMGC.IMAGES_PER_PAGE     = 4 ;
  IMGC.ASPECT_RATIO        = 1.5 ;
  IMGC.THUMB_MICROFICATION = 4 ;
  IMGC.BASE_IMAGE_WIDTH    = IMGC.DEF_IMAGE_WIDTH ;
  IMGC.BASE_IMAGE_HEIGHT   = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO);
  IMGC.setBorderSize() ;
  IMGC.paintImages();
 }, false);
 Event.observe($('fourXfour'),    'click', function()	
 {
  IMGC.IMAGES_PER_ROW      = 4   ;
  IMGC.IMAGES_PER_COL      = 4;
  IMGC.IMAGES_PER_PAGE     = IMGC.IMAGES_PER_ROW * IMGC.IMAGES_PER_COL;
  IMGC.ASPECT_RATIO        = 1.5 ;
  IMGC.THUMB_MICROFICATION = 4 ;
  IMGC.BASE_IMAGE_WIDTH    = IMGC.DEF_IMAGE_WIDTH ;
  IMGC.BASE_IMAGE_HEIGHT   = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ;
  IMGC.setBorderSize() ;
  IMGC.paintImages();
 }, false);
 Event.observe($('twoXone'),      'click', function()	
 {
  IMGC.IMAGES_PER_ROW      = 2   ;
  IMGC.IMAGES_PER_COL      = 1;
  IMGC.IMAGES_PER_PAGE     = IMGC.IMAGES_PER_ROW * IMGC.IMAGES_PER_COL;
  IMGC.ASPECT_RATIO        = 1.5 ;
  IMGC.THUMB_MICROFICATION = 4 ;
  IMGC.BASE_IMAGE_WIDTH    = IMGC.DEF_IMAGE_WIDTH ;
  IMGC.BASE_IMAGE_HEIGHT   = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ;
  IMGC.setBorderSize() ;
  IMGC.paintImages();
 }, false);

 IMGC.loadingProgress('hidden') ;

}, false);

//__________________________________________________________________________________________________________________________________
IMGC.loadBatch = function ()	
{
 var getTitles = new Ajax.Request('services/filesTitlesBatch.lis', // Load titles first, because they are 
 	 		         {                                 // used by the IMGC.processImageList
 	 		          method: 'get',                   // which fires later on
 			          parameters: '', 
 			          onComplete: IMGC.processTitlesList // <-- call back function
 			         });
 var getFiles  = new Ajax.Request('services/filesBatch.lis', 
 	 		         {
 	 		          method: 'get', 
 			          parameters: '', 
 			          onComplete: IMGC.processImageList // <-- call back function
 			         });
}
//__________________________________________________________________________________________________________________________________
IMGC.getImageList = function ()	
{
 var getTitles = new Ajax.Request(IMGC.IMAGE_LIST_TITLES, // Load titles first, because they are 
 	 		         {                        // used by the IMGC.processImageList
 	 		          method: 'get',          // which fires later on
 			          parameters: '', 
 			          onComplete: IMGC.processTitlesList // <-- call back function
 			         });
 var getFiles  = new Ajax.Request(IMGC.IMAGE_LIST_URL, 
 	 		         {
 	 		          method: 'get', 
 			          parameters: '', 
 			          onComplete: IMGC.processImageList // <-- call back function
 			         });
}
//__________________________________________________________________________________________________________________________________
IMGC.updateIMGC = function (source)	
{
 if( !source )
 {
  source = IMGC.lastSource ;
  if( source == "" ) return ;
 } else {
  IMGC.lastSource = source ;
 }
 
 var url = WebLib.getApplicationURL2();
 url = url + 
       '/Request?RequestID=updateIMGCPlots&MEFolder=' +
       source ;

 // Ajax request to get back the list of ME that will be displayed by the call-back function
 var getMEURLS = new Ajax.Request(url,                    
 	 		         {			  
 	 		          method: 'get',	  
 			          parameters: '', 
 			          onComplete: IMGC.processIMGCPlots // <-- call-back function
 			         });
}
//__________________________________________________________________________________________________________________________________
IMGC.updateAlarmsIMGC = function (path)	
{
 var url      = WebLib.getApplicationURL2();
 url         += "/Request?";
 queryString  = 'RequestID=PlotHistogramFromPath';
 queryString += '&Path='   + path;
 queryString += '&width='  + IMGC.BASE_IMAGE_WIDTH +
                '&height=' + IMGC.BASE_IMAGE_HEIGHT ;
 queryString += '&histotype=qtest';
 url         += queryString;
 
 var getMEURLS = new Ajax.Request(url,                    
 	 		         {			  
 	 		          method: 'get',	  
 			          parameters: '', 
 			          onComplete: IMGC.processIMGCPlots // <-- call-back function
 			         });
}
//__________________________________________________________________________________________________________________________________
IMGC.processIMGCPlots = function (ajax)	
{
 var imageURLs;
 var url = WebLib.getApplicationURL2();

 try	 
 { 
  imageURLs = ajax.responseText.split(/\s+/) ;
 } catch(e) {
  alert('[IMGC::processIMGCPlots()] Image URLs list load failed. Reason: '+e.message);
  return 0;	  
 }

 date = new Date() ; // This is extremely important: adding a date to the QUERY_STRING of the
                     // URL, forces the browser to reload the picture even if the Plot, Folder
		     // and canvas size are the same (e.g. by clicking twice the same plot)
		     // The reload is forced because the browser is faked to think that the
		     // URL is ALWAYS different from an already existing one.
		     // This was rather tricky... (Dario)
 date = date.toString() ;
 date = date.replace(/\s+/g,"_") ;

 var canvasW             = window.innerWidth * IMGC.GLOBAL_RATIO ;
 IMGC.DEF_IMAGE_WIDTH    = parseInt(canvasW);
 IMGC.BASE_IMAGE_WIDTH   = IMGC.DEF_IMAGE_WIDTH;
 IMGC.BASE_IMAGE_HEIGHT  = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ;
 IMGC.THUMB_IMAGE_WIDTH  = IMGC.BASE_IMAGE_WIDTH  / IMGC.IMAGES_PER_ROW;
 IMGC.THUMB_IMAGE_HEIGHT = IMGC.BASE_IMAGE_HEIGHT / IMGC.IMAGES_PER_COL;
 
 var theFolder = imageURLs[0] ;
 for( var i=1; i<imageURLs.length-1; i++)
 {
  imageURLs[i-1] = url                                    + 
                   "/Request?RequestID=getIMGCPlot&Plot=" + 
		   imageURLs[i]  	 		  +
		   "&Folder="    	 		  +
		   theFolder    	 		  +
        	   '&canvasW='  	 		  +
        	   IMGC.BASE_IMAGE_WIDTH 		  +
        	   '&canvasH='           		  +
        	   IMGC.BASE_IMAGE_HEIGHT 		  +
        	   "&Date="               		  +
        	   date;
 }

 $('imageCanvas').imageList     = imageURLs;
 $('imageCanvas').titlesList    = imageURLs;
 $('imageCanvas').current_start = 0;
 IMGC.PATH_TO_PICTURES = "" ; 
 IMGC.computeCanvasSize() ;
}
//__________________________________________________________________________________________________________________________________
IMGC.processImageList = function (ajax)	
{
 var imageList;

 try	 
 {
  imageList = eval('(' + ajax.responseText + ')');
 } catch(e) {
  alert('[IMGC::processImageList()] Image list load failed.');
  return 0;	  
 }
 
 $('imageCanvas').imageList     = imageList;
 $('imageCanvas').current_start = 0;

 IMGC.computeCanvasSize() ;
}

//__________________________________________________________________________________________________________________________________
IMGC.processTitlesList = function (ajax)	
{
 var titlesList;

 try	 
 {
  titlesList = eval('(' + ajax.responseText + ')');
 } catch(e) {
  alert('[IMGC::processTitlesList()] Image titles list load failed.');
  titlesList = "No text collected" ;
  return 0;	  
 }

 $('imageCanvas').titlesList = titlesList;
}
//__________________________________________________________________________________________________________________________________
IMGC.setBorderSize = function ()
{
 var theBorder  = $('canvasBorder') ;
 var theBorderS = "width: "  + IMGC.BASE_IMAGE_WIDTH  + "px; " +
                  "height: " + IMGC.BASE_IMAGE_HEIGHT + "px" ;
 theBorder.setAttribute("style",theBorderS) ;
}

//__________________________________________________________________________________________________________________________________
IMGC.resize = function ()
{ 
 IMGC.changeSize('=') ;
}
//__________________________________________________________________________________________________________________________________
IMGC.computeCanvasSize = function ()
{ 
 var canvasW             = window.innerWidth * IMGC.GLOBAL_RATIO ;
 IMGC.DEF_IMAGE_WIDTH    = parseInt(canvasW);
 IMGC.BASE_IMAGE_WIDTH   = IMGC.DEF_IMAGE_WIDTH;
 IMGC.BASE_IMAGE_HEIGHT  = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ;
 IMGC.THUMB_IMAGE_WIDTH  = IMGC.BASE_IMAGE_WIDTH  / IMGC.IMAGES_PER_ROW;
 IMGC.THUMB_IMAGE_HEIGHT = IMGC.BASE_IMAGE_HEIGHT / IMGC.IMAGES_PER_COL;
 IMGC.setBorderSize() ;
 IMGC.paintImages();
}

//__________________________________________________________________________________________________________________________________
IMGC.changeSize = function (direction)
{
 if(       direction == '-')
 {
  IMGC.BASE_IMAGE_WIDTH  = parseInt(IMGC.BASE_IMAGE_WIDTH * .9)                ; // Scale down by 10%
  IMGC.BASE_IMAGE_HEIGHT = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ; // Keep fixed aspect ratio
 } else if(direction == '+')  {
  IMGC.BASE_IMAGE_WIDTH  = parseInt(IMGC.BASE_IMAGE_WIDTH / .9)                ; // Scale upby 10%
  IMGC.BASE_IMAGE_HEIGHT = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ; // Keep fixed aspect ratio
 } else if(direction == '=')  { 				    
  IMGC.BASE_IMAGE_WIDTH  = IMGC.DEF_IMAGE_WIDTH                                ; // Restore default size
  IMGC.BASE_IMAGE_HEIGHT = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ; // Keep fixed aspect ratio
 } else if(direction == 'M')  {
  var headerH     = IMGC.getStyleValue('header',     'height') ; 
  var headerTextH = IMGC.getStyleValue('headerText', 'height') ; 
  var controlsH   = IMGC.getStyleValue('controls',   'height') ;
  var borderH     = IMGC.getStyleValue('border',     'height') ; 
//   alert("\ncontrolsH  : "+controlsH+
//         "\nheaderH    : "+headerH+
//         "\nheaderT    : "+headerT+
//         "\nheaderTextH: "+headerTextH+
// 	"\nborderH    : "+borderH+
// 	"\nwindow.innerHeight: "+window.innerHeight) ;
  var dimX = window.innerWidth  - 100;
  var dimY = window.innerHeight - (controlsH+headerH+headerTextH) ;
  IMGC.BASE_IMAGE_WIDTH   = dimX ;
  IMGC.BASE_IMAGE_HEIGHT  = dimY ;
//   alert("\ndimX: "+dimX+
//         "\ndimY: "+dimY) ;
 }
  
 IMGC.setBorderSize() ;
 IMGC.paintImages();
}

//__________________________________________________________________________________________________________________________________
IMGC.getStyleValue = function (tagName,styleType) 
{
 var style = $(tagName).getStyle(styleType) ;
 var parts = style.split("px") ;
 return parseInt(parts[0]) ;
}

//__________________________________________________________________________________________________________________________________
IMGC.repaintUponResize = function()
{
 IMGC.updateIMGC() ;
}
//__________________________________________________________________________________________________________________________________
IMGC.paintImages = function ()	
{
// new Effect.Fade($('demo-all')) ;
 
 var imageList   = $('imageCanvas').imageList;
 var titlesList  = $('imageCanvas').titlesList;
 var imageCanvas = $('imageCanvas');

 IMGC.THUMB_IMAGE_WIDTH   = IMGC.BASE_IMAGE_WIDTH  / IMGC.IMAGES_PER_ROW;
 IMGC.THUMB_IMAGE_HEIGHT  = IMGC.BASE_IMAGE_HEIGHT / IMGC.IMAGES_PER_COL;

 while(imageCanvas.hasChildNodes())
 {
  imageCanvas.removeChild(imageCanvas.firstChild);
 }
 	 
 for(var i = imageCanvas.current_start; i < imageList.length && i < IMGC.IMAGES_PER_PAGE + imageCanvas.current_start; i++) 
 {
  var img	     = document.createElement('img');
  img.src	     = IMGC.PATH_TO_PICTURES   + imageList[i];
  img.style.width    = IMGC.THUMB_IMAGE_WIDTH  + 'px';
  img.style.height   = IMGC.THUMB_IMAGE_HEIGHT + 'px';
  img.image_index    = i - imageCanvas.current_start ;		  
  img.style.position = 'absolute';
  img.style.cursor   = 'pointer';
  img.style.left     = img.image_index % IMGC.IMAGES_PER_ROW * IMGC.THUMB_IMAGE_WIDTH + 'px';
  img.style.top      = parseInt(img.image_index / IMGC.IMAGES_PER_COL) * IMGC.THUMB_IMAGE_HEIGHT + 'px';
  img.style.zIndex   = 1;
  img.style.opacity  = IMGC.INACTIVE_OPACITY;
  img.style.filter   = 'alpha(opacity=' + IMGC.INACTIVE_OPACITY * 100 + ')';

  imageCanvas.appendChild(img);
 }

 var markup	       = imageCanvas.innerHTML;
 imageCanvas.innerHTML = '';
 imageCanvas.innerHTML = markup;

 for(var i = 0; i < imageCanvas.childNodes.length; i++) 
 {	 
  var img	  = imageCanvas.childNodes[i];
  img.image_index = i;  	  
  img.imageNumber = i + imageCanvas.current_start;		  
  img.slide_fx    = new Fx.Styles(img, {duration: 300, transition: Fx.Transitions.expoOut});
  img.opacity_fx  = new Fx.Styles(img, {duration: 300, transition: Fx.Transitions.quadOut});
 
  Event.observe(img, 'mouseover', function()	  
  {
//   $('traceRegion').slideDown = new Effect.SlideDown($('demo-all')) ;
   var imageList  = $('imageCanvas').imageList;
   var element    = window.event ? window.event.srcElement : this;
   var thisPlot   = $('imageCanvas').titlesList[element.imageNumber].split("Plot=") ;
   var plotName   = "" ;
   var plotFolder = "" ;
   var temp ;
   try 
   {
    temp          = thisPlot[1].split(/&/g) ;
    plotName      = temp[0] ;
    plotFolder    = temp[1] ;
    temp          = plotFolder.split("Folder=" );
    plotFolder    = temp[1] ;
    plotFolder.replace(/Collector\/(FU\d+)\/Tracker/,"$1/") ;
    plotFolder.replace(/Collector/,"") ;
    plotFolder.replace(/Collated/,"") ;
    plotFolder.replace(/Tracker/,"") ;
   } catch(e) {}
   
//   $('traceRegion').value = "Plot: " + element.imageNumber + "  " + plotName + " | " +  plotFolder;

   element.opacity_fx.clearTimer();
   element.opacity_fx.custom({
 	                     'opacity' : [parseFloat(element.style.opacity), 1]
                             });
 
   element.slide_fx.clearTimer();
   element.slide_fx.custom({
 	   		   'width'  : [element.offsetWidth,  IMGC.THUMB_IMAGE_WIDTH  * 1.1],
 	   		   'height' : [element.offsetHeight, IMGC.THUMB_IMAGE_HEIGHT * 1.1],
 	   		   'left'   : [element.offsetLeft, element.offsetLeft - IMGC.THUMB_IMAGE_WIDTH * .05],
 	   		   'top'    : [element.offsetTop,  element.offsetTop  - IMGC.THUMB_IMAGE_WIDTH * .05]
                           });
   $('imgTitle').value = element.imageNumber + "]  " + plotFolder + "  |  " +  plotName;;
  }, false);
 
  Event.observe(img, 'mouseout', function()	  
  {
   var element = window.event ? window.event.srcElement : this;;
   element.opacity_fx.clearTimer();
   element.opacity_fx.custom({
 	                     'opacity' : [parseFloat(element.style.opacity), IMGC.INACTIVE_OPACITY]
                             });
 
   element.slide_fx.clearTimer();
   element.slide_fx.custom({
 	   		   'width'  : [element.offsetWidth,  IMGC.THUMB_IMAGE_WIDTH],
 	   		   'height' : [element.offsetHeight, IMGC.THUMB_IMAGE_HEIGHT],
 	   		   'left'   : [element.offsetLeft, element.image_index % IMGC.IMAGES_PER_ROW * IMGC.THUMB_IMAGE_WIDTH],
 	   		   'top'    : [element.offsetTop, parseInt(element.image_index / IMGC.IMAGES_PER_ROW) * IMGC.THUMB_IMAGE_HEIGHT]
                           });
   $('imgTitle').value = "" ;
  }, false);
 	 	  
  Event.observe(img, 'click', IMGC.handleImageClick, false);
 }
}

//__________________________________________________________________________________________________________________________________
IMGC.updatePage = function (direction)	
{
 if(       direction == 'next')
 {
  $('imageCanvas').current_start += IMGC.IMAGES_PER_PAGE;	 
 } else if(direction == 'previous') {
  $('imageCanvas').current_start -= IMGC.IMAGES_PER_PAGE;	 
 } else if(direction == 'first')    {
  $('imageCanvas').current_start  = 0;	 
 } else if(direction == 'last')     {
  var numberOfPages = parseInt($('imageCanvas').imageList.length / IMGC.IMAGES_PER_PAGE) ;
  var lastPage      = parseInt(numberOfPages * IMGC.IMAGES_PER_PAGE  ) ;
  $('imageCanvas').current_start  = lastPage;
 }
 
 if($('imageCanvas').current_start > $('imageCanvas').imageList.length)	 
 {
  alert('[IMGC::updatePage()] No more images!');
  $('imageCanvas').current_start -= IMGC.IMAGES_PER_PAGE;	  
  return 0;			  
 }

 if($('imageCanvas').current_start < 0) 
 {
  alert('[IMGC::updatePage()] You\'re already at the beginning!');
  $('imageCanvas').current_start += IMGC.IMAGES_PER_PAGE;	  
  return 0;			  
 }
 	 
 IMGC.paintImages();
}

//__________________________________________________________________________________________________________________________________
IMGC.handleImageClick = function ()	
{
 var element = window.event ? window.event.srcElement : this;

 element.slide_fx   = new Fx.Styles(element, {duration: 300, transition: Fx.Transitions.expoOut});
 element.opacity_fx = new Fx.Styles(element, {duration: 300, transition: Fx.Transitions.quadOut});

 if(element.offsetWidth != IMGC.BASE_IMAGE_WIDTH)	 
 {
//  new Effect.Fade($('demo-all')) ;
  element.style.zIndex = 2;
 
  element.slide_fx.clearTimer();
  element.slide_fx.custom({
 	  'width'  : [element.offsetWidth,  IMGC.BASE_IMAGE_WIDTH],
 	  'height' : [element.offsetHeight, IMGC.BASE_IMAGE_HEIGHT],
 	  'left'   : [element.offsetLeft, 0],
 	  'top'    : [element.offsetTop, Math.max(0, Math.min($('imageCanvas').offsetHeight - IMGC.BASE_IMAGE_HEIGHT, element.offsetTop - (IMGC.BASE_IMAGE_HEIGHT / 2) + (IMGC.BASE_IMAGE_HEIGHT / (IMGC.THUMB_MICROFICATION * 2))))]
  });
 
  element.opacity_fx.clearTimer();
  element.opacity_fx.custom({
			  'opacity' : [IMGC.INACTIVE_OPACITY, 1]
			  });
			     
  for(var i = 0; i < element.parentNode.childNodes.length; i++)   
  {
   var sibling = element.parentNode.childNodes[i];
 
   if(sibling != element) 
    {
     sibling.style.zIndex = 1;
     if( sibling.offsetWidth  != IMGC.THUMB_IMAGE_WIDTH || 
 	 sibling.offsetHeight != IMGC.THUMB_IMAGE_HEIGHT )       
     {
 	 sibling.slide_fx.clearTimer();
 	 sibling.slide_fx.custom({
 	    	 'width'  : [sibling.offsetWidth,  IMGC.THUMB_IMAGE_WIDTH],
 	    	 'height' : [sibling.offsetHeight, IMGC.THUMB_IMAGE_HEIGHT],
 	    	 'left'   : [sibling.offsetLeft, sibling.image_index % IMGC.IMAGES_PER_ROW * IMGC.THUMB_IMAGE_WIDTH],
 	    	 'top'    : [sibling.offsetTop, parseInt(sibling.image_index / IMGC.IMAGES_PER_ROW) * IMGC.THUMB_IMAGE_HEIGHT]
 	 });
     }
   }
  }
 } else	{
  element.style.zIndex = 1;
 
  element.slide_fx.clearTimer();
  element.slide_fx.custom({
 	  'width'  : [element.offsetWidth,  IMGC.THUMB_IMAGE_WIDTH],
 	  'height' : [element.offsetHeight, IMGC.THUMB_IMAGE_HEIGHT],
 	  'left'   : [element.offsetLeft, element.image_index % IMGC.IMAGES_PER_ROW * IMGC.THUMB_IMAGE_WIDTH],
 	  'top'    : [element.offsetTop, parseInt(element.image_index / IMGC.IMAGES_PER_ROW) * IMGC.THUMB_IMAGE_HEIGHT]					  
  });
			     
 }


}	
//__________________________________________________________________________________________________________________________________
IMGC.changeHeaderOpacity = function (state)	
{
 var element = $('headerText') ;
 element.slide_fx   = new Fx.Styles(element, {duration: 200, transition: Fx.Transitions.expoOut});
 element.opacity_fx = new Fx.Styles(element, {duration: 400, transition: Fx.Transitions.quadOut});

 element.opacity_fx.clearTimer();
 if(        state == 'mouseover' )
 {
  element.opacity_fx.custom({
			    'opacity' : [1, IMGC.INACTIVE_OPACITY]
			    });
 } else if (state == 'mouseout') {
  element.opacity_fx.custom({
			    'opacity' : [IMGC.INACTIVE_OPACITY, 1]
			    });
 }
}
//__________________________________________________________________________________________________________________________________
IMGC.loadingProgress = function (state)	
{
 var element = $('progressIcon') ;
 element.opacity_fx = new Fx.Styles(element, {duration: 1000, transition: Fx.Transitions.quadOut});
 element.opacity_fx.clearTimer();
 if( state == 'visible')
 {
  element.opacity_fx.custom({
			    'opacity' : [0, 1]
			    });
 } else {
  element.opacity_fx.custom({
			    'opacity' : [1, 0]
			    });
 }
}
//__________________________________________________________________________________________________________________________________
IMGC.selectedIMGCItems = function ()	
{
 var url       = WebLib.getApplicationURL2();
 var imageURLs = new Array();
 var selection = document.getElementsByName("selected") ;
 date = new Date() ; // This is extremely important: adding a date to the QUERY_STRING of the
                     // URL, forces the browser to reload the picture even if the Plot, Folder
		     // and canvas size are the same (e.g. by clicking twice the same plot)
		     // The reload is forced because the browser is faked to think that the
		     // URL is ALWAYS different from an already existing one.
		     // This was rather tricky... (Dario)
 date = date.toString() ;
 date = date.replace(/\s+/g,"_") ;

 var canvasW             = window.innerWidth * IMGC.GLOBAL_RATIO ;
 IMGC.DEF_IMAGE_WIDTH    = parseInt(canvasW);
 IMGC.BASE_IMAGE_WIDTH   = IMGC.DEF_IMAGE_WIDTH;
 IMGC.BASE_IMAGE_HEIGHT  = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ;
 IMGC.THUMB_IMAGE_WIDTH  = IMGC.BASE_IMAGE_WIDTH  / IMGC.IMAGES_PER_ROW;
 IMGC.THUMB_IMAGE_HEIGHT = IMGC.BASE_IMAGE_HEIGHT / IMGC.IMAGES_PER_COL;
 
 for( var i=0; i<selection.length; i++)
 {
  if( selection[i].checked )
  {
   var qs = url                                    + 
            "/Request?RequestID=getIMGCPlot&Plot=" + 
	    selection[i].value  		   +
	    "&Folder="  			   +
	    selection[i].getAttribute("folder")    +
            '&canvasW=' 			   +
            IMGC.BASE_IMAGE_WIDTH		   +
            '&canvasH=' 			   +
            IMGC.BASE_IMAGE_HEIGHT		   +
            "&Date="				   +
            date;
   imageURLs.push(qs) ;
  }
 }

 $('imageCanvas').imageList     = imageURLs;
 $('imageCanvas').titlesList    = imageURLs;
 $('imageCanvas').current_start = 0;
 IMGC.PATH_TO_PICTURES = "" ; 
 IMGC.computeCanvasSize() ;
}
//__________________________________________________________________________________________________________________________________
IMGC.clearSelection = function ()	
{
 var selection = document.getElementsByName("selected") ;
 for( var i=0; i<selection.length; i++)
 {
  if( selection[i].checked )
  {
   selection[i].checked = false ;
  }
 }
 
 IMGC.selectedIMGCItems() ;
}
