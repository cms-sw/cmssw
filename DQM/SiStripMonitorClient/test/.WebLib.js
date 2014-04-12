//-------------------------------------------------------------------------|
//  A few basic actions to get the url of the application Web Page without |
//  asking the Web Server. These methods are used by a number of other     |
//  java acripts.                                                           |
//                                                                         |
//  Also the entry point for all necessary javascripts for the             |
//  DQM application                                                        |
//-------------------------------------------------------------------------|
var WebLib = {};

  WebLib.http_request      = null;
  WebLib.view_all_contents = true;
//--------------------------------------------------------------------------                           // Get the url location of the application web page removing the
// last part without asking the server...
//
WebLib.getApplicationURL = function() 
{
  try {
   var url = window.location.href;
   // remove the cgi request from the end of the string
   var index = url.indexOf("?");
   if (index >= 0) {
     url = url.substring(0, index);
   }
   // remove the trailing '/' from the end of the string
   index = url.lastIndexOf("/");
   url = url.substring(0, index);
   return url;
  } catch (errorMessage) {
   alert("[WebLib.getApplicationURL] Exeuction/syntax error: " + errorMessage ) ;
  }
}
//--------------------------------------------------------------------------                           //  Get the url of the application webpage with correct LID 
//   without asking the server...
//
WebLib.getApplicationURLWithLID = function() 
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
   var s0          = (url.lastIndexOf(":")+1);
   var s1          = url.lastIndexOf("/");
   var port_number = url.substring(s0, s1);
   if (port_number == "40000") {
    url += "urn:xdaq-application:lid=27/moduleWeb?module=SiStripAnalyser&";
   } else if (port_number == "1972") {
     url += "urn:xdaq-application:lid=15/Request?";
   }
   return url;
 } catch (errorMessage) {
   alert("[WebLib.getApplicationURL2] Exeuction/syntax error: " + errorMessage ) ;
 }   
}
//--------------------------------------------------------------------------                           // Get the context url location of the application web page, i.e.
// the string up to the port number
WebLib.getContextURL = function()
{
 try 
 {
  var app_url = WebLib.getApplicationURL();
  var index   = app_url.lastIndexOf("/");
  return app_url.substring(0, index);
 } catch (errorMessage) {
  alert("[WebLib.getContextURL] Exeuction/syntax error: " + errorMessage ) ;
 }
}
//--------------------------------------------------------------------------                           //
//  -- This function submits a generic request in the form of a url
//  -- and calls the receiver_function when the state of the request
//  -- changes.
//
WebLib.makeRequest = function(url, receiver_function) 
{
 try 
 {
   WebLib.http_request = null;
   if (window.XMLHttpRequest) { 
     WebLib.http_request = new XMLHttpRequest();
     if (WebLib.http_request.overrideMimeType){
       WebLib.http_request.overrideMimeType('text/xml');
     }
   } else if (window.ActiveXObject) { 
     WebLib.http_request = new ActiveXObject("Msxml2.XMLHTTP"); 
     if (WebLib.http_request == null) { 
       WebLib.http_request = new ActiveXObject("Microsoft.XMLHTTP"); 
     } 
   } 
   if (WebLib.http_request) { 
     WebLib.initReq("GET", url, true, receiver_function); 
   } else { 
     alert('Giving up :( Cannot create an XMLHTTP instance');
   } 
 } catch (errorMessage) {
   alert("[WebLib.makeRequest] Exeuction/syntax error: " + errorMessage ) ;
 }
}
//--------------------------------------------------------------------------                           //
// -- Initialize a request object that is already constructed 
//
WebLib.initReq = function(reqType, url, bool, respHandle)
{ 
  try { 	
    // Specify the function that will handle the HTTP response 
    if (respHandle != null)  WebLib.http_request.onreadystatechange = respHandle;
    WebLib.http_request.open(reqType, url, bool); 

    // if the reqType parameter is POST, then the 
    // 5th argument to the function is the POSTed data 
    if (reqType.toLowerCase() == "post") { 
     WebLib.http_request.setRequestHeader("Content-Type", 
           "application/x-www-form-urlencoded; charset=UTF-8"); 
      WebLib.http_request.send(arguments[4]); 
    } else { 
      WebLib.http_request.send(null); 
    } 
  } 
  catch (errv) { 
    alert ( 
        "The application cannot contact " + 
        "the server at the moment. " + 
        "Please try again in a few seconds.\\n" + 
        "Error detail: " + errv.message); 
  } 
}
document.write('<script src="SERVED_DIRECTORY_URL/js_files/RequestHistos.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/CommonActions.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/tab-view.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/context-menu.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/folder-tree-static.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/SlideShow.js"><\/script>');

