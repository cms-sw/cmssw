
var ClientActions = {} ;

//___________________________________________________________________________________
// Subscribe All MEs 
ClientActions.SubscribeAll = function () {
  var queryString = "RequestID=SubscribeAll";
  var url = WebLib.getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString; 
  
  WebLib.makeRequest(url, WebLib.dummy);
  
  WebLib.enableButtons("listMECommand") ;    
}

//___________________________________________________________________________________
//
// Setup Quality Test
//
ClientActions.SetupQualityTest = function() { // Unused?
  var queryString = "RequestID=SetupQTest";
  var url = WebLib.getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString; 
  
  WebLib.makeRequest(url, WebLib.dummy);     
}

//___________________________________________________________________________________
//
// Check Quality Test Results
//
ClientActions.CheckQualityTestResults = function() {
  var queryString = "RequestID=CheckQTResults";
  var url = WebLib.getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString; 
  
  WebLib.makeRequest(url, WebLib.dummy);     
}

//___________________________________________________________________________________
//
// Create Summary
//
ClientActions.CreateSummary = function() {
  var queryString = "RequestID=CreateSummary";
  var url = WebLib.getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString; 
  
  WebLib.makeRequest(url, WebLib.dummy);     
}

//___________________________________________________________________________________
//
// Same MEs in a file
//
ClientActions.SaveToFile = function() {
  var queryString = "RequestID=SaveToFile";
  var url = WebLib.getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString;   
  WebLib.makeRequest(url, WebLib.dummy);     
}

//___________________________________________________________________________________
// eisenach watburg gotha
// Create Tracker Map
//
ClientActions.CreateTrackerMap = function() {
  var queryString = "RequestID=CreateTkMap";
  var obj = document.getElementById("create_tkmap");

  var url = WebLib.getApplicationURL2();
  url += "/Request?";
  url += queryString; 
  var obj = document.getElementById("monitoring_element_list");
  var sname =  obj.options[obj.selectedIndex].value;
  url += '&MEName='+sname;
   
  WebLib.makeRequest(url, ClientActions.ReadResponseAndOpenTkMap);
}

//___________________________________________________________________________________
//
// Create Tracker Map
//
ClientActions.OpenTrackerMap = function() { // Unused?
  var queryString = "RequestID=OpenTkMap";

  var url = WebLib.getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString;
   
  WebLib.makeRequest(url, ClientActions.ReadResponseAndOpenTkMap); 
}

//___________________________________________________________________________________
// check the response and open tracker map
ClientActions.ReadResponseAndOpenTkMap = function() 
{
  if (WebLib.http_request.readyState == 4) 
  {
    if (WebLib.http_request.status == 200) 
    {
      try 
      {
       var win = window.open("TrackerMapFrame.html",
             		     "trackerMapWindow"    ,
             		     "menubar	= no,  "   +
             		     "location  = no,  "   +
             		     "resizable = no,  "   +
             		     "scrollbars= yes, "   +
             		     "titlebar  = yes, "   +
             		     "status	= yes, "   +
             		     "left	=   0, "   +
             		     "top	=   0, "   +
             		     "height	= 700, "   +
             		     "width	= 1280 ")  ;
       win.moveTo(0,0) ;
       win.focus();	       
      } catch (err) {
        alert ("[ClientActions.ReadResponseAndOpenTkMap] ERROR: " + err.message);
      }
    } else {
      alert("[ClientActions.ReadResponseAndOpenTkMap] ERROR: "+WebLib.http_request.readyState+", "+WebLib.http_request.status);
    }
  }
}

//___________________________________________________________________________________
//
// Check Quality Test Results
//
ClientActions.CollateME = function() {
  var queryString = "RequestID=CollateME";
  var url = WebLib.getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString; 
  
  WebLib.makeRequest(url, WebLib.dummy);     
}
