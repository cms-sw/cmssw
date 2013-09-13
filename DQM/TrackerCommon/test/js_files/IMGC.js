//_______________________________________________________________________
// Author: D. Menasce                                                    |
//                                                                       |
// A dynamic canvas to display pictures on a customizibale grid (nxm).   |
// Each picture responds to mouse actions:                               |
// - when the mouse hovers on a picture, this is highlighted (it gets    |
//   slightly enlarged)                                                  |
// - when the user clicks on a picture, this is magnified to bring it to |
//   full resolution                                                     |
//_______________________________________________________________________|

// Crate an IMGC namespace (all local variables or functions belonging to this namespace
// have the IMGC prefixed to them)
var IMGC = {} ;

// Initialize local variables
IMGC.IMAGE_LIST_URL      = '/images/filesList.lis';
IMGC.IMAGE_LIST_TITLES   = '/images/filesTitles.lis';
IMGC.PATH_TO_PICTURES    = '/images/';
IMGC.IMAGES_PER_ROW      = 2;
IMGC.IMAGES_PER_COL      = 2;
IMGC.IMAGES_PER_PAGE     = IMGC.IMAGES_PER_ROW * IMGC.IMAGES_PER_COL;
IMGC.THUMB_MICROFICATION = 4;
IMGC.INACTIVE_OPACITY    = 0.7;
IMGC.DEF_IMAGE_WIDTH     = 600;
IMGC.ASPECT_RATIO        = 1.5 ;
IMGC.BASE_IMAGE_WIDTH    = IMGC.DEF_IMAGE_WIDTH;
IMGC.BASE_IMAGE_HEIGHT   = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ;
IMGC.THUMB_IMAGE_WIDTH   = IMGC.BASE_IMAGE_WIDTH  / IMGC.IMAGES_PER_ROW;
IMGC.THUMB_IMAGE_HEIGHT  = IMGC.BASE_IMAGE_HEIGHT / IMGC.IMAGES_PER_COL;
IMGC.GLOBAL_RATIO        = .6 ;
IMGC.lastSource          = "" ;

//__________________________________________________________________________________________________________________________________
// When the HTML document that requests this script is fully loaded, this functions is
// called and proper initialization occurs.
// Initialization will create a canvas with all the necessary buttons and decorations
// and will trigger an Ajax request to the server to get the list of available
// plots. It will further subscribe buttons to their appropriate call-back function
// to provide users with navigation tools.
//
Event.observe(window, 'load', function()	
{
 IMGC.initialize();

 IMGC.loadingProgress('visible') ;

 IMGC.getImageList() ;
 
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
// Create the canvas buttons and decorations. This is implemented by dynamically adding
// HTML statements to the current document as childrens of the 'theCanvas' <DIV> element
IMGC.initialize = function ()	
{
 try
 {
  var tmp = $('theCanvas').getAttribute("style") ;
 } catch(errorMessage) {
  alert("[IMGC.js::IMGC.initialize()]\nNo <DIV> element found with ID='theCanvas' in the current HTML file") ;
  return ;
 }
 var theCanvas               = $('theCanvas') ;
 var theFieldset             = document.createElement("fieldset") ;
 var theLegend               = document.createElement("legend") ;
 var theMainContainer        = document.createElement("div") ;
 var theControls             = document.createElement("div") ;
 var theNormalSize           = document.createElement("button") ;
 var theSpan1                = document.createElement("span") ;
 var theSmallerSize          = document.createElement("button") ;
 var theSpan2                = document.createElement("span") ;
 var theFirstPage            = document.createElement("button") ;
 var theSpan3                = document.createElement("span") ;
 var thePreviousPage         = document.createElement("button") ;
 var theSpan4                = document.createElement("span") ;
 var theNextPage             = document.createElement("button") ;
 var theSpan5                = document.createElement("span") ;
 var theLastPage             = document.createElement("button") ;
 var theSpan6                = document.createElement("span") ;
 var theLargerSize           = document.createElement("button") ;
 var theSpan7                = document.createElement("span") ;
 var theMaximizeSize         = document.createElement("button") ;
 var theSpan8                = document.createElement("span") ;
 var thefourXfour            = document.createElement("button") ;
 var theSpan9                = document.createElement("span") ;
 var thetwoXtwo              = document.createElement("button") ;
 var theSpan10               = document.createElement("span") ;
 var thetwoXone              = document.createElement("button") ;
 var theSpan11               = document.createElement("span") ;
 var theoneXone              = document.createElement("button") ;
 var thePar1                 = document.createElement("p") ;
 var theImgTitleDiv          = document.createElement("div") ;
 var theImgTitle             = document.createElement("input") ;
 var thePar2                 = document.createElement("p") ;
 var theCanvasBorder         = document.createElement("div") ;
 var theImageCanvas          = document.createElement("div") ;

 theLegend.textContent       = " Dynamic drawing canvas " ;
 theNormalSize.textContent   = "=" ;
 theSpan1.textContent        = " | " ;
 theSmallerSize.textContent  = "-" ;
 theSpan2.textContent        = " | " ;
 theFirstPage.textContent    = "<<" ;
 theSpan3.textContent        = " | " ;
 thePreviousPage.textContent = "<" ;
 theSpan4.textContent        = " | " ;
 theNextPage.textContent     = ">" ;
 theSpan5.textContent        = " | " ;
 theLastPage.textContent     = ">>" ;
 theSpan6.textContent        = " | " ;
 theLargerSize.textContent   = "+" ;
 theSpan7.textContent        = " | " ;
 theMaximizeSize.textContent = "M" ;
 theSpan8.textContent        = " | " ;
 thefourXfour.textContent    = "4x4" ;
 theSpan9.textContent        = " | " ;
 thetwoXtwo.textContent      = "2x2" ;
 theSpan10.textContent       = " | " ;
 thetwoXone.textContent      = "2x1" ;
 theSpan11.textContent       = " | " ;
 theoneXone.textContent      = "1x1" ;

 theFieldset.setAttribute(     "id",	"theCanvasField" );
 theFieldset.setAttribute(     "style", "background: #225587; width: 95%; height: 100%;" );
 theLegend.setAttribute(       "id",	"theLegend" );
 theLegend.setAttribute(       "style",	"background: #225587; font-family: Verdana, Arial; font-size: 12px; color: #ffb400;" );
 theMainContainer.setAttribute("id",    "mainContainer") ;
 theMainContainer.setAttribute("align", "center") ;
 theNormalSize.setAttribute(   "type",  "submit") ;
 theNormalSize.setAttribute(   "class", "controlButton") ;
 theNormalSize.setAttribute(   "id",	"normalSize") ;
 theNormalSize.setAttribute(   "value", "=") ;
 theSmallerSize.setAttribute(  "type",  "submit") ;
 theSmallerSize.setAttribute(  "class", "controlButton") ;
 theSmallerSize.setAttribute(  "id",	"smallerSize") ;
 theSmallerSize.setAttribute(  "value", "-") ;
 theFirstPage.setAttribute(    "type",  "submit") ;
 theFirstPage.setAttribute(    "class", "controlButton") ;
 theFirstPage.setAttribute(    "id",	"firstPage") ;
 theFirstPage.setAttribute(    "value", "&lt;&lt;") ;
 thePreviousPage.setAttribute( "type",  "submit") ;
 thePreviousPage.setAttribute( "class", "controlButton") ;
 thePreviousPage.setAttribute( "id",	"previousPage") ;
 thePreviousPage.setAttribute( "value", "&lt;") ;
 theNextPage.setAttribute(     "type",  "submit") ;
 theNextPage.setAttribute(     "class", "controlButton") ;
 theNextPage.setAttribute(     "id",	"nextPage") ;
 theNextPage.setAttribute(     "value", "&gt;") ;
 theLastPage.setAttribute(     "type",  "submit") ;
 theLastPage.setAttribute(     "class", "controlButton") ;
 theLastPage.setAttribute(     "id",    "lastPage") ;
 theLastPage.setAttribute(     "value", "&gt;&gt;") ;
 theLargerSize.setAttribute(   "type",  "submit") ;
 theLargerSize.setAttribute(   "class", "controlButton") ;
 theLargerSize.setAttribute(   "id",    "largerSize") ;
 theLargerSize.setAttribute(   "value", "+") ;
 theMaximizeSize.setAttribute( "type",  "submit") ;
 theMaximizeSize.setAttribute( "class", "controlButton") ;
 theMaximizeSize.setAttribute( "id",	"maximizeSize") ;
 theMaximizeSize.setAttribute( "value", "M") ;
 thefourXfour.setAttribute(    "type",  "submit") ;
 thefourXfour.setAttribute(    "class", "controlButton") ;
 thefourXfour.setAttribute(    "id",	"fourXfour") ;
 thefourXfour.setAttribute(    "value", "4x4") ;
 thetwoXtwo.setAttribute(      "type",  "submit") ;
 thetwoXtwo.setAttribute(      "class", "controlButton") ;
 thetwoXtwo.setAttribute(      "id",	"twoXtwo") ;
 thetwoXtwo.setAttribute(      "value", "2x2") ;
 thetwoXone.setAttribute(      "type",  "submit") ;
 thetwoXone.setAttribute(      "class", "controlButton") ;
 thetwoXone.setAttribute(      "id",    "twoXone") ;
 thetwoXone.setAttribute(      "value", "2x1") ;
 theoneXone.setAttribute(      "type",  "submit") ;
 theoneXone.setAttribute(      "class", "controlButton") ;
 theoneXone.setAttribute(      "id",	"oneXone") ;
 theoneXone.setAttribute(      "value", "1x1") ;
 theImgTitleDiv.setAttribute(  "id",	"imgTitleDiv") ;
 theImgTitle.setAttribute(     "id",	"imgTitle") ;
 theImgTitle.setAttribute(     "class",	"inputText") ;
 theImgTitle.setAttribute(     "type",	"text") ;
 theImgTitle.setAttribute(     "value",	"") ;
 theCanvasBorder.setAttribute( "id",	"canvasBorder") ;
 theImageCanvas.setAttribute(  "id",	"imageCanvas") ;
 theImageCanvas.setAttribute(  "style",	"position: relative") ;

 theCanvas.appendChild(theFieldset) ;
 theFieldset.appendChild(theLegend) ;
 theFieldset.appendChild(theMainContainer) ;
 theMainContainer.appendChild(theControls) ;
 theControls.appendChild(theNormalSize) ;
 theControls.appendChild(theSpan1) ;
 theControls.appendChild(theSmallerSize) ;
 theControls.appendChild(theSpan2) ;
 theControls.appendChild(theFirstPage) ;
 theControls.appendChild(theSpan3) ;
 theControls.appendChild(thePreviousPage) ;
 theControls.appendChild(theSpan4) ;
 theControls.appendChild(theNextPage) ;
 theControls.appendChild(theSpan5) ;
 theControls.appendChild(theLastPage) ;
 theControls.appendChild(theSpan6) ;
 theControls.appendChild(theLargerSize) ;
 theControls.appendChild(theSpan7) ;
 theControls.appendChild(theMaximizeSize) ;
 theControls.appendChild(theSpan8) ;
 theControls.appendChild(thefourXfour) ;
 theControls.appendChild(theSpan9) ;
 theControls.appendChild(thetwoXtwo) ;
 theControls.appendChild(theSpan10) ;
 theControls.appendChild(thetwoXone) ;
 theControls.appendChild(theSpan11) ;
 theControls.appendChild(theoneXone) ;
 theControls.appendChild(thePar1) ;
 theMainContainer.appendChild(theImgTitleDiv) ;
 theImgTitleDiv.appendChild(theImgTitle) ;
 theMainContainer.appendChild(thePar2) ;
 theMainContainer.appendChild(theCanvasBorder ) ;
 theCanvasBorder.appendChild(theImageCanvas) ;

}

//__________________________________________________________________________________________________________________________________
// Function called once during initialization: takes the list of inital pictures from the 
// IMGC.IMAGE_LIST_URL vector and issues a request to the web server to provide those for
// first time display
IMGC.getImageList = function ()	
{
 try 
 {
  var url = IMGC.getURL() ;
  var urlTitleList = url + IMGC.IMAGE_LIST_TITLES;
  var urlImageList = url + IMGC.IMAGE_LIST_URL ;
  var getTitles = new Ajax.Request(urlTitleList,	   // Load titles first, because they are
  				  {			   // used by the IMGC.processImageList
  				   method: 'get',	   // which fires later on
  				   parameters: '', 
  				   onComplete: IMGC.processTitlesList // <-- call back function
  				  });
  var getFiles  = new Ajax.Request(urlImageList, 
  				  {
  				   method: 'get', 
  				   parameters: '', 
  				   onComplete: IMGC.processImageList  // <-- call back function
  				  });
 } catch(errorMessage) {
  alert("[IMGC.js::IMGC.getImageList()]\nExecution/syntax error: " + error.errorMessage ) ;
 }
}

//__________________________________________________________________________________________________________________________________
// Internal Utility Function: returns the full path to the current HTML document with the document file 
// name stripped from it
IMGC.getURL = function()
{
 try 
 {
  var url = window.location.href;
  var list = url.split(/\//) ;
  var match = list[list.length-1].match(/.*\.html/) ;
  if( match != null )
  {
   url = url.replace(match,"") ;
  }
  return url;
 } catch(errorMessage) {
  alert("[IMGC.js::IMGC.getURL()]\nExecution/syntax error: " + error.errorMessage ) ;
 }
}

//__________________________________________________________________________________________________________________________________
// This is were user requests are forwarded to the web server: input is the full path-name to the
// DQM directory containing the required Monitoring Elements. The QUERY STRING of the request
// contains the full path-name: the server will respond with a list of ME names that will be 
// used to prepare hyperlink addresses on the canvas. The answer of the web server is dealt by
// IMGC.processIMGCPlots Ajax callback function.
IMGC.updateIMGC = function (source)	
{
 if( !source )
 {
  source = IMGC.lastSource ;
  if( source == "" ) return ;
 } else {
  IMGC.lastSource = source ;
 }
 
 var url = IMGC.getApplicationURL();
 url = url + 
       'RequestID=updateIMGCPlots&MEFolder=' +
       source;

 // Ajax request to get back the list of ME that will be displayed by the call-back function
 var getMEURLS = new Ajax.Request(url,                    
 	 		         {			  
 	 		          method: 'get',	  
 			          parameters: '', 
 			          onComplete: IMGC.processIMGCPlots // <-- call-back function
 			         });
}

//__________________________________________________________________________________________________________________________________
// Unused function (it remains here as a reference for possibile future uses)
IMGC.updateAlarmsIMGC = function (path)	
{
 var url      = IMGC.getApplicationURL();
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
// This is an Ajax callback function. It is activated when the web server responds to the request of 
// providing a list of ME available in a particular DQM folder. The response is packaged by the server
// as a vector of strings containing the ME names. This list is explored and a new vector is created
// with each element containing the complete QUERY-STRING to ask the server to provide the corresponding
// ME plot in the form of a PNG chunk of data. Once this vector is ready (imageURLs) the canvas is 
// refreshed with a call to IMGC.computeCanvasSize().
IMGC.processIMGCPlots = function (ajax)	
{
 var imageURLs;
 var url = IMGC.getApplicationURL();
// var url = "";

 try	 
 { 
  imageURLs = ajax.responseText.split(/\s+/) ;
 } catch(errorMessage) {
  alert('[IMGC.js::IMGC.processIMGCPlots()]\nImage URLs list load failed. Reason: '+error.errorMessage);
  return 0;	  
 }


 try	 
 { 
  date = new Date() ; // This is extremely important: adding a date to the QUERY_STRING of the
  		      // URL, forces the browser to reload the picture even if the Plot, Folder
  		      // and canvas size are the same (e.g. by clicking twice the same plot)
  		      // The reload is forced because the browser is faked to think that the
  		      // URL is ALWAYS different from an already existing one.
  		      // This was rather tricky... (Dario)
  date = date.toString() ;
  date = date.replace(/\s+/g,"_") ;

  var canvasW		  = window.innerWidth * IMGC.GLOBAL_RATIO ;
  IMGC.DEF_IMAGE_WIDTH    = parseInt(canvasW);
  IMGC.BASE_IMAGE_WIDTH   = IMGC.DEF_IMAGE_WIDTH;
  IMGC.BASE_IMAGE_HEIGHT  = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ;
  IMGC.THUMB_IMAGE_WIDTH  = IMGC.BASE_IMAGE_WIDTH  / IMGC.IMAGES_PER_ROW;
  IMGC.THUMB_IMAGE_HEIGHT = IMGC.BASE_IMAGE_HEIGHT / IMGC.IMAGES_PER_COL;
 
  var theFolder = imageURLs[0] ;
  var tempURLs   = new Array() ;
  var tempTitles = new Array() ;
  for( var i=1; i<imageURLs.length-1; i++)
  {
   var fullPath = theFolder + "/" + imageURLs[i] ;
   tempURLs[i-1] = url 				   + 
  		    "RequestID=GetIMGCImage&Path=" + 
  		    fullPath			   +
  		    '&width=' 			   +
  		    IMGC.BASE_IMAGE_WIDTH		   +
  		    '&height=' 			   +
  		    IMGC.BASE_IMAGE_HEIGHT		   +
  		    "&Date="				   +
  		    date;
    tempTitles[i-1] =theFolder + "|" + imageURLs[i]  
  }

  $('imageCanvas').imageList	 = tempURLs;
  $('imageCanvas').titlesList	 = tempTitles;
  $('imageCanvas').current_start = 0;
  IMGC.PATH_TO_PICTURES = "" ; 
  IMGC.computeCanvasSize() ;
 } catch(errorMessage) {
  alert('[IMGC.js::IMGC.processIMGCPlots()]\nExecution/syntax error: '+error.errorMessage);
 }
}
//__________________________________________________________________________________________________________________________________
// Internal Utility Function: returns the URL needed to communicate with web server. A request to the
// server is composed by this URL plus an optional QUERY-STRING.
IMGC.getApplicationURL = function()
{
 try 
 {
  var url = window.location.href;
  // remove the cgi request from the end of the string
  var index = url.indexOf("?");
  if (index >= 0)
  {
    url = url.substring(0, index);
  }

  index = url.lastIndexOf("temporary");
  url   = url.substring(0, index);

  // add the cgi request
  var s0	  = (url.lastIndexOf(":")+1);
  var s1	  = url.lastIndexOf("/");
  var port_number = url.substring(s0, s1);
  if (port_number == "40000") {
   url += "urn:xdaq-application:lid=27/moduleWeb?module=SiStripAnalyser&";
  } else if (port_number == "1972") {
    url += "urn:xdaq-application:lid=15/Request?";
  }
  return url;
 } catch(errorMessage) {
  alert("[IMGC.js::IMGC.getApplicationURL()]\nExecution/syntax error: " + error.errorMessage ) ;
 }
}
//__________________________________________________________________________________________________________________________________
// This is an Ajax callback function. It is activated when the web server responds to the request of 
// providing the list of images to display in the startup canvas
// (this is called only once by IMGC.getImageList).
IMGC.processImageList = function (ajax)	
{
 var imageList;
 try	 
 {
  imageList = eval('(' + ajax.responseText + ')');
 } catch(errorMessage) {
  alert('[IMGC.js::IMGC.processImageList()]\nImage list load failed. Error: '+error.errorMessage);
  return 0;	  
 }

 $('imageCanvas').imageList     = imageList;
// $('imageCanvas').ChandraURLs   = imageList;
 $('imageCanvas').current_start = 0;

 IMGC.computeCanvasSize() ;
}

//__________________________________________________________________________________________________________________________________
IMGC.processTitlesList = function (ajax)	
// This is an Ajax callback function. It is activated when the web server responds to the request of 
// providing the list of image titles to display in the startup canvas
// (this is called only once by IMGC.getImageList).
{
 var titlesList;

 try	 
 {
  titlesList = eval('(' + ajax.responseText + ')');
 } catch(errorMessage) {
  alert('[IMGC.js::IMGC.processTitlesList()]\nImage titles list load failed. Error: '+error.errorMessage);
  titlesList = "No text collected" ;
  return 0;	  
 }

 $('imageCanvas').titlesList = titlesList;
}

//__________________________________________________________________________________________________________________________________
// Internal Utility Function: refreshes the size attributes of the canvas (these may change upon a
// user-generated resize of the browser's window)
IMGC.setBorderSize = function ()
{
 var theBorder  = $('canvasBorder') ;
 var theBorderS = "width: "  + IMGC.BASE_IMAGE_WIDTH  + "px; " +
                  "height: " + IMGC.BASE_IMAGE_HEIGHT + "px" ;
 theBorder.setAttribute("style",theBorderS) ;
}

//__________________________________________________________________________________________________________________________________
// Obsolete function: was just a switchyard to the IMGC.changeSize method specialized to the
// same-size resize
IMGC.resize = function ()
{ 
 IMGC.changeSize('=') ;
}

//__________________________________________________________________________________________________________________________________
// Refreshes the internal variables with the current size of the browser's window (which might have
// changed after a user's resize) and calls IMGC.paintImages() to refresh the canvas with new plots
// provided by the web server with the newly adjusted pixel resolution.
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
// Callback function associated with some of the IMGC canvas control button's.  
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
  var headerH     = IMGC.getStyleValue('header',     'height') ;                 // Scales up to completely fill 
  var headerTextH = IMGC.getStyleValue('headerText', 'height') ;                 // the available area of the canvas
  var controlsH   = IMGC.getStyleValue('controls',   'height') ;
  var borderH     = IMGC.getStyleValue('border',     'height') ; 
  var dimX = window.innerWidth  - 100;
  var dimY = window.innerHeight - (controlsH+headerH+headerTextH) ;
  IMGC.BASE_IMAGE_WIDTH   = dimX ;
  IMGC.BASE_IMAGE_HEIGHT  = dimY ;
 }
  
 IMGC.setBorderSize() ;
 IMGC.paintImages();
}

//__________________________________________________________________________________________________________________________________
// Internal Utility Function: returns the numerical part of the style attribute (specified by styleType)
// for the HTML element specified by tagName
IMGC.getStyleValue = function (tagName,styleType) 
{
 var style = $(tagName).getStyle(styleType) ;
 var parts = style.split("px") ;
 return parseInt(parts[0]) ;
}

//__________________________________________________________________________________________________________________________________
// Callback function associated to the onresize action in the HTML body of the calling document
IMGC.repaintUponResize = function()
{
 IMGC.updateIMGC() ;
}
//__________________________________________________________________________________________________________________________________
// This is were the canvas gets filled with plot content. Each plot is contained is an <IMG> element
// which is created here on the fly (one for every plot in the current nxm grid). Before each <IMG>
// element is created, previously existing ones are removed. The style of each <IMG> element is then 
// initialized to a somewhat reduced opacity and callback functions are dynamically registered: these
// control the behaviour of the image when the mouse hovers on them or the user clicks one.
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
  imageCanvas.removeChild(imageCanvas.firstChild); // Remove pre-existing <IMG> elements in the current canvas
 }
 	 
 // Create a new <IMG> element for each picture and define it's initial style 
 for(var i = imageCanvas.current_start; i < imageList.length && i < IMGC.IMAGES_PER_PAGE + imageCanvas.current_start; i++) 
 {
  var img	     = document.createElement('img');
//  img.src	     = IMGC.getURL()           + imageList[i];
  img.src	     = imageList[i];
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

 // Associate a transition behaviour to each plot in the canvas and register appropriate callback functions 
 // to deal with mouse events
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
   var thisPlot   = $('imageCanvas').titlesList[element.imageNumber];
//   var thisPlot   = $('imageCanvas').titlesList[element.imageNumber].split("Plot=") ;
//   var plotName   = "" ;
//   var plotFolder = "" ;
//   var temp ;
//   try 
//   {
//    temp          = thisPlot[1].split(/&/g) ;
//    plotName      = temp[0] ;
//    plotFolder    = temp[1] ;
//    temp          = plotFolder.split("Folder=" );
//    plotFolder    = temp[1] ;
//    plotFolder.replace(/Collector\/(FU\d+)\/Tracker/,"$1/") ;
//    plotFolder.replace(/Collector/,"") ;
//    plotFolder.replace(/Collated/,"") ;
//    plotFolder.replace(/Tracker/,"") ;
//   } catch(e) {}
   
//   $('traceRegion').value = "Plot: " + element.imageNumber + "  " + plotName + " | " +  plotFolder;

   element.opacity_fx.clearTimer();
   element.opacity_fx.custom({
 	                     'opacity' : [parseFloat(element.style.opacity), 1]
                             });
 
   element.slide_fx.clearTimer();
   element.slide_fx.custom({
 	   		   'width'  : [element.offsetWidth,  IMGC.THUMB_IMAGE_WIDTH  * 1.05],
 	   		   'height' : [element.offsetHeight, IMGC.THUMB_IMAGE_HEIGHT * 1.05],
 	   		   'left'   : [element.offsetLeft, element.offsetLeft - IMGC.THUMB_IMAGE_WIDTH * .01],
 	   		   'top'    : [element.offsetTop,  element.offsetTop  - IMGC.THUMB_IMAGE_WIDTH * .01]
                           });
//   $('imgTitle').value = element.imageNumber + "]  " + plotFolder + "  |  " +  plotName;;
   $('imgTitle').value = element.imageNumber + "]  " + thisPlot;

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
// Handles the display of a specific page in the canvas: this is a callback function associated to the canvas
// control buttons 
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
  alert('[IMGC.js::IMGC.updatePage()]\nNo more images!');
  $('imageCanvas').current_start -= IMGC.IMAGES_PER_PAGE;	  
  return 0;			  
 }

 if($('imageCanvas').current_start < 0) 
 {
  alert('[IMGC.js::IMGC.updatePage()]\nYou\'re already at the beginning!');
  $('imageCanvas').current_start += IMGC.IMAGES_PER_PAGE;	  
  return 0;			  
 }
 	 
 IMGC.paintImages();
}

//__________________________________________________________________________________________________________________________________
// This is were the dynamic behaviour of the canvas is implemented. The picture clicked by the
// user fills up the entire canvas.
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
// This is an obsolete function (it's here just for future reference)
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
// This is were the progress bar is made visible or invisible: this works only if the calling HTML 
// page contains a <DIV> element identified by ID=progressIcon. Usually this <DIV> element contains
// an <IMG> element with a spinning animated gif 
IMGC.loadingProgress = function (state)	
{
 try
 {
  var tmp = $('progressIcon').getAttribute("style") ;
 } catch(errorMessage) {
  return ;
 }
 
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
// This functions is triggered each time a user clicks on a check-button connected to a ME. 
// It explores the checked status of each check-button and prepares an appropriate QUERY-STRING
// for the display of the plot.
IMGC.selectedIMGCItems = function ()	
{
 var url       = IMGC.getApplicationURL();
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
// Clears the selection of check-boxes and, as a side effect, all plots are removed fro the canvas
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
//__________________________________________________________________________________________________________________________________
// This is an Ajax callback function. It is activated when the web server responds to the request of 
// providing a list of ME available in a particular DQM folder. The response is packaged by the server
// as a vector of strings containing the ME names. This list is explored and a new vector is created
// with each element containing the complete QUERY-STRING to ask the server to provide the corresponding
// ME plot in the form of a PNG chunk of data. Once this vector is ready (imageURLs) the canvas is 
// refreshed with a call to IMGC.computeCanvasSize() after a time delay.
IMGC.processImageURLs = function (ajax)	
{
 var imageURLs;
 var url = IMGC.getApplicationURL();

 try	 
 { 
  imageURLs = ajax.responseText.split(/\s+/) ;
 } catch(errorMessage) {
  alert('[IMGC.js::IMGC.processIMGCPlots()]\nImage URLs list load failed. Reason: '+error.errorMessage);
  return 0;	  
 }


 try	 
 { 

  var canvasW		  = window.innerWidth * IMGC.GLOBAL_RATIO ;
  IMGC.DEF_IMAGE_WIDTH    = parseInt(canvasW);
  IMGC.BASE_IMAGE_WIDTH   = IMGC.DEF_IMAGE_WIDTH;
  IMGC.BASE_IMAGE_HEIGHT  = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO) ;
  IMGC.THUMB_IMAGE_WIDTH  = IMGC.BASE_IMAGE_WIDTH  / IMGC.IMAGES_PER_ROW;
  IMGC.THUMB_IMAGE_HEIGHT = IMGC.BASE_IMAGE_HEIGHT / IMGC.IMAGES_PER_COL;

  date = new Date() ; // This is extremely important: adding a date to the QUERY_STRING of the
                     // URL, forces the browser to reload the picture even if the Plot, Folder
		     // and canvas size are the same (e.g. by clicking twice the same plot)
		     // The reload is forced because the browser is faked to think that the
		     // URL is ALWAYS different from an already existing one.
		     // This was rather tricky... (Dario)
  date = date.toString() ;
  date = date.replace(/\s+/g,"_") ;
 
  var theFolder = imageURLs[0] ;
  var tempURLs   = new Array() ;
  var tempTitles = new Array() ;
  for( var i=1; i<imageURLs.length-1; i++)
  {
   var fullPath = theFolder + "/" + imageURLs[i] ;
   tempURLs[i-1] = url + "RequestID=GetIMGCImage&Path=" + fullPath 
                       + "&Date=" + date;
   tempTitles[i-1] =theFolder + "|" + imageURLs[i]  
  }

  $('imageCanvas').imageList	 = tempURLs;
  $('imageCanvas').titlesList	 = tempTitles;
  $('imageCanvas').current_start = 0;
  IMGC.PATH_TO_PICTURES = "" ; 
   setTimeout('IMGC.computeCanvasSize()',10000) ;
 } catch(errorMessage) {
  alert('[IMGC.js::IMGC.processIMGCPlots()]\nExecution/syntax error: '+error.errorMessage);
 }
}
