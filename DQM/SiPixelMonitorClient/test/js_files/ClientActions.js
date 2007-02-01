// Subscribe All MEs 
function SubscribeAll() {
  var queryString = "RequestID=SubscribeAll";
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
//
// Create Tracker Map
//
function CreateTrackerMap() {
  var queryString = "RequestID=CreateTkMap";
  var obj = document.getElementById("create_tkmap");

  var url = getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString;
   
  makeRequest(url, dummy);
 
  setTimeout('OpenTrackerMap()', 500000);   
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
function ReadResponseAndOpenTkMap() {

  if (http_request.readyState == 4) {
    if (http_request.status == 200) {
      try {
        var doc = http_request.responseXML;
        var root = doc.documentElement;
        var rows = root.getElementsByTagName('Response');
        if ( rows.length == 1) { 
          var name  = rows[0].childNodes[0].nodeValue;
          if (name == "Successful" ) {            
             var win = window.open('embedded_svg.html');
             win.focus();            
          } else {
            alert(" Creation of Tracker Map Failed !! ");	
          }
        }
      }
      catch (err) {
        alert ("Error detail: " + err.message);
      }
    }
    else {
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
