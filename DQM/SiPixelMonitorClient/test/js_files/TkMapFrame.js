window.onload=function(){
  TkMapFrame.SetupTkMap();
}
var TkMapFrame = {} 
TkMapFrame.SetupTkMap = function() 
{
  var url = TkMapFrame.getURLWithLID();
  var queryString = "RequestID=GetTkMap";
  url	     += queryString; 
  document.getElementById("tkmap_area").src = url;
}
TkMapFrame.getURLWithLID = function() 
{
 try 
 { 
   var url = window.location.href;
   // remove the cgi request from the end of the string
   var index = url.indexOf("?");	
   if (index >= 0) {
     url = url.substring(0, index);
   }
   index = url.lastIndexOf("temporary");
   url   = url.substring(0, index);

   // add the cgi request
   var s0	   = (url.lastIndexOf(":")+1);
   var s1	   = url.lastIndexOf("/");
   var port_number = url.substring(s0, s1);
   url += "urn:xdaq-application:lid=27/moduleWeb?module=SiPixelEDAClient&";
   return url;
 } catch (errorMessage) {
   alert("[TkMapFrame.getURLWithLID] Exeuction/syntax error: " + errorMessage );
 }   
}
TkMapFrame.requestMPlot = function(detid) { 
   var parea  = parent.plotArea;
   var canvas = parea.IMGC;
   var queryString = "RequestID=PlotTkMapHistogram";
   queryString+= "&ModId=" + detid;
   canvas.computeCanvasSize();
   queryString += '&width='+canvas.BASE_IMAGE_WIDTH+
		  '&height='+canvas.BASE_IMAGE_HEIGHT;
   canvas.IMAGES_PER_ROW      = 2;
   canvas.IMAGES_PER_COL      = 2;
   canvas.IMAGES_PER_PAGE     = canvas.IMAGES_PER_ROW * canvas.IMAGES_PER_COL;

   var url =  TkMapFrame.getURLWithLID();

   url += queryString;

   var getMEURLS = new parent.plotArea.Ajax.Request(url,
	       {
		  method: 'get',
		  parameters: '',
		  onComplete: canvas.processImageURLs // <-- call-back function
	       });
}
