
var ClientActions = {} ;

//___________________________________________________________________________________
// Subscribe All MEs 
ClientActions.SubscribeAll = function () {
  var queryString = "RequestID=SubscribeAll";
  var url = WebLib.getApplicationURL2();
  //url = url + "/Request?";
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
  //url = url + "/Request?";
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
  //url = url + "/Request?";
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
  //url = url + "/Request?";
  url = url + queryString; 
  
  WebLib.makeRequest(url, WebLib.dummy);     
}

//___________________________________________________________________________________
//
// Same MEs in a file
//
ClientActions.SaveToFile = function() {
  //alert("HALLO!" + err.message);
  var queryString = "RequestID=SaveToFile";
  var url = WebLib.getApplicationURL2();
  //url = url + "/Request?";
  url = url + queryString;   
  WebLib.makeRequest(url, WebLib.dummy);     
}

//___________________________________________________________________________________
// eisenach watburg gotha
// Create Tracker Map
//
ClientActions.CreateTrackerMap = function() 
{
  IMGC.loadingProgress("visible") ;	
  
  var queryString = "RequestID=CreateTkMap";
  var obj         = document.getElementById("create_tkmap");

  var url         = WebLib.getApplicationURL2();
  url 		 += queryString;
  var obj   	  = document.getElementById("monitoring_element_list");
  var sname 	  = obj.options[obj.selectedIndex].value;
  obj       	  = document.getElementById("TKMapContentType");
  var stype 	  = obj.options[obj.selectedIndex].value;
  url 		 += '&MEName='    + sname;
  url 		 += '&TKMapType=' + stype;
   
  WebLib.makeRequest(url, ClientActions.ReadResponseAndOpenTkMap);
}

//___________________________________________________________________________________
//
// Create Tracker Map
//
ClientActions.OpenTrackerMap = function() { // Unused?
  var queryString = "RequestID=OpenTkMap";

  var url = WebLib.getApplicationURL2();
  //url = url + "/Request?";
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
       ClientActions.smallDelay(5000) ;
       var doc  = WebLib.http_request.responseXML;	  
       var root = doc.documentElement;			  
       var dets = root.getElementsByTagName("Response") ;  
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
             		     "width	= 1150 ")  ;
       win.moveTo(0,0) ;
       win.focus();	       
       IMGC.loadingProgress("hidden") ;	
      } catch (err) {
        alert ("[ClientActions.ReadResponseAndOpenTkMap] ERROR: " + err.message);
      }
    } else {
      alert("[ClientActions.ReadResponseAndOpenTkMap] ERROR: "+WebLib.http_request.readyState+", "+WebLib.http_request.status);
    }
  }
 }

 //------------------------------------------------------------------------------------------
 ClientActions.smallDelay = function(millis)
 {
   var inizio     = new Date();
   var inizioint  = inizio.getTime();
   var intervallo = 0;
   while(intervallo<millis)
   {
     var fine     = new Date();
     var fineint  = fine.getTime();
     intervallo   = fineint-inizioint;
   }
 }
//___________________________________________________________________________________
//
// Check Quality Test Results
//
ClientActions.DumpModIds = function() {
  var queryString = "RequestID=dumpModIds";
  var url = WebLib.getApplicationURL2();
  url = url + queryString; 
  
  WebLib.makeRequest(url, WebLib.dummy);     
}
//___________________________________________________________________________________
//
// Request Plot from layout
//
ClientActions.RequestPlotFromLayout = function()
{
  try{
   var queryString;
   var url = WebLib.getApplicationURL2();
   queryString = "RequestID=PlotHistogramFromLayout";
   url += queryString;
   WebLib.makeRequest(url,WebLib.dummy);
  }catch(e){
   alert(e.message);
  }
}
//___________________________________________________________________________________//
//
// Request Error Overview Plot
//
ClientActions.RequestErrorOverviewPlot = function()
{
  try{
   var queryString;
   var url = WebLib.getApplicationURL2();
   //url += "/Request?";
   queryString = "RequestID=PlotErrorOverviewHistogram";
   url += queryString;
   WebLib.makeRequest(url,WebLib.dummy);
  }catch(e){
   alert(e.message);
  }
}
//___________________________________________________________________________________
