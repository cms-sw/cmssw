// Subscribe All MEs 
function SubscribeAll() {
  var queryString = "RequestID=SubscribeAll";
  var url = getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString; 
  
  makeRequest(url, dummy);
  
  enableButtons("listMECommand") ;    
}
//
// Setup Quality Test
//
function SetupQualityTest() {
  var queryString = "RequestID=SetupQTest";
  var url = getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString; 
  
  makeRequest(url, dummy);     
}
//
// Check Quality Test Results
//
function CheckQualityTestResults() {
  var queryString = "RequestID=CheckQTResults";
  var url = getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString; 
  
  makeRequest(url, dummy);     
}
//
// Create Summary
//
function CreateSummary() {
  var queryString = "RequestID=CreateSummary";
  var url = getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString; 
  
  makeRequest(url, dummy);     
}
//
// Same MEs in a file
//
function SaveToFile() {
  var queryString = "RequestID=SaveToFile";
  var url = getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString;   
  makeRequest(url, dummy);     
}
// eisenach watburg gotha
// Create Tracker Map
//
function CreateTrackerMap() {
  var queryString = "RequestID=CreateTkMap";
  var obj = document.getElementById("create_tkmap");

  var url = getApplicationURL2();
  url += "/Request?";
  url += queryString; 
  var obj = document.getElementById("monitoring_element_list");
  var sname =  obj.options[obj.selectedIndex].value;
  url += '&MEName='+sname;
   
//  makeRequest(url, dummy);
  makeRequest(url, ReadResponseAndOpenTkMap);
 
//  setTimeout('OpenTrackerMap()', 5000);   
}
//
// Create Tracker Map
//
function OpenTrackerMap() {
  var queryString = "RequestID=OpenTkMap";

  var url = getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString;
   
  makeRequest(url, ReadResponseAndOpenTkMap); 
}

// check the response and open tracker map
function ReadResponseAndOpenTkMap() 
{
  if (http_request.readyState == 4) 
  {
    if (http_request.status == 200) 
    {
      try 
      {
//        var doc = http_request.responseXML;
//        var root = doc.documentElement;
//        var rows = root.getElementsByTagName('Response');
//        alert("[ClientActions.js"+arguments.callee.name+"] "+"rows.length "+rows.length) ; 
//        if ( rows.length == 1) 
//	{ 
//          var name  = rows[0].childNodes[0].nodeValue;
//         alert("[ClientActions.js"+arguments.callee.name+"] "+"name        "+name) ; 
//          if (name == "Successful" ) 
//	  {            
             var win = window.open("TrackerMapFrame.html",
	                           "trackerMapWindow"    ,
                                   "menubar   = no,  "   +
                                   "location  = no,  "   +
                                   "resizable = no,  "   +
                                   "scrollbars= yes, "   +
                                   "titlebar  = yes, "   +
                                   "status    = yes, "   +
                                   "left      =   0, "   +
                                   "top       =   0, "   +
                                   "height    = 700, "   +
                                   "width     = 1280 ")  ;
	     win.moveTo(0,0) ;
             win.focus();            
//          } else {
//            alert(" Creation of Tracker Map Failed !! ");	
//          }
//        }
      } catch (err) {
        alert ("Error detail: " + err.message);
      }
    } else {
      alert("FillFileList:  ERROR:"+http_request.readyState+", "+http_request.status);
    }
  }
}
//
// Check Quality Test Results
//
function CollateME() {
  var queryString = "RequestID=CollateME";
  var url = getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString; 
  
  makeRequest(url, dummy);     
}
