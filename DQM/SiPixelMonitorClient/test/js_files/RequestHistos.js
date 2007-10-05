
var RequestHistos = {} ;

//___________________________________________________________________________________
RequestHistos.RequestModuleHistoList = function() { // Unused? 
  var queryString;
  var url = WebLib.getApplicationURL2();
  //url += "/Request?";
  queryString = "RequestID=SingleModuleHistoList";
  url += queryString; 
  WebLib.makeRequest(url, RequestHistos.FillModuleHistoList);     
}

//___________________________________________________________________________________
RequestHistos.RequestMEList = function(what) {
  var queryString;
  var url = WebLib.getApplicationURL2();
  //url += "/Request?";
  queryString = "RequestID=GetMEList";
  url += queryString; 
  if( what == 'CB' )
  { 
   WebLib.makeRequest(url, RequestHistos.FillMEList); 
  } else {
   WebLib.makeRequest(url, RequestHistos.ReturnMEList); 
  }    
}

//___________________________________________________________________________________
RequestHistos.RequestSummaryHistoList = function() {
  IMGC.loadingProgress('visible') ;
  var queryString;
  var url = WebLib.getApplicationURL2();
  //url += "/Request?";
  queryString = "RequestID=SummaryHistoList";
  var obj = document.getElementById("structure_name");
  var sname =  obj.options[obj.selectedIndex].value;
  queryString += '&StructureName='+sname;
  url += queryString; 
  WebLib.makeRequest(url, RequestHistos.FillSummaryHistoList);     
}

//___________________________________________________________________________________
RequestHistos.RequestModuleTree = function() {
  IMGC.loadingProgress('visible') ;
  var queryString;
  var url = WebLib.getApplicationURL2();
//alert("RequestModuleTree:" + url);
  //url += "/Request?";
  queryString = "RequestID=ModuleHistoList";
  var obj = document.getElementById("structure_for_module");
  var sname =  obj.options[obj.selectedIndex].value;
  queryString += '&StructureName='+sname;
  url += queryString; 
//alert("RequestModuleTree:" + url);
  WebLib.makeRequest(url, RequestHistos.FillModuleTree);     
}

//___________________________________________________________________________________
RequestHistos.RequestAlarmList = function() {
  var queryString;
  var url = WebLib.getApplicationURL2();
  //url += "/Request?";
  queryString = "RequestID=AlarmList";
  var obj = document.getElementById("structure_for_alarm");
  var sname =  obj.options[obj.selectedIndex].value;
  queryString += '&StructureName='+sname;
  url += queryString; 
  WebLib.makeRequest(url, RequestHistos.FillAlarmList);     
}

//___________________________________________________________________________________
RequestHistos.FillModuleHistoList = function() {
  if (WebLib.http_request.readyState == 4) {
    if (WebLib.http_request.status == 200) {
      try {


        var doc = WebLib.http_request.responseXML;
        var root = doc.documentElement;
        
        // Module Number select box
        var aobj = document.getElementById("module_numbers");
        aobj.options.length = 0;
        
        var mrows = root.getElementsByTagName('ModuleNum');
        for (var i = 0; i < mrows.length; i++) {
          var mnum  = mrows[i].childNodes[0].nodeValue;
          var aoption = new Option(mnum, mnum);
          try {
            aobj.add(aoption, null);
          }
          catch (e) {
            aobj.add(aoption, -1);
          }
        }

        // Histogram  select box
        var bobj = document.getElementById("histolistarea");
        bobj.options.length = 0;

        var hrows = root.getElementsByTagName('Histo');
        for (var j = 0; j < hrows.length; j++) {
          var name  = hrows[j].childNodes[0].nodeValue;
          var boption = new Option(name, name);
          try {
            bobj.add(boption, null);
          }
          catch (e) {
            bobj.add(boption, -1);
          }
        }
      }
      catch (err) {
        alert ("Error detail: " + err.message); 
      }
    }
  }
}

//___________________________________________________________________________________
RequestHistos.FillMEList = function() {
  if (WebLib.http_request.readyState == 4) 
  {
    if (WebLib.http_request.status == 200) 
    {
      try 
      {

        var doc = WebLib.http_request.responseXML;
        var root = doc.documentElement;
        
        // Module Number select box
        // Histogram  select box
        var bobj = document.getElementById("monitoring_element_list");
        bobj.options.length = 0;

        var hrows = root.getElementsByTagName('Histo');
        for (var j = 0; j < hrows.length; j++) 
	{
          var name     = hrows[j].childNodes[0].nodeValue;
          var htype    = hrows[j].getAttribute("type");
	  var fullName = htype + " - " + name ;  
          var boption = new Option(fullName, name);
          try 
	  {
            bobj.add(boption, null);
          }
          catch (e) 
	  {
            bobj.add(boption, -1);
          }
        }
        WebLib.enableButtons("UpdateTrackerMap") ;    
      }
      catch (err) {
        alert ("[RequestHistos.js::FillMEList()] Error detail: " + err.message); 
      }
    }
  }
}

//___________________________________________________________________________________
RequestHistos.ReturnMEList = function() 
{
  if (WebLib.http_request.readyState == 4) 
  {
    if (WebLib.http_request.status == 200) 
    {
      try 
      {
        var doc   = WebLib.http_request.responseXML;
        var root  = doc.documentElement;
        var theME = document.getElementsByName("MEReference");
        var hrows = root.getElementsByTagName('Histo');

        for (var j= 0; j < hrows.length; j++) 
	{
	  theME[j].setAttribute("value",hrows[j].childNodes[0].nodeValue) ;
        }
      } catch (err) {
        alert ("[RequestHistos.js::ReturnMEList()] Error detail: " + err.message); 
      }
    }
  }
}

//___________________________________________________________________________________
RequestHistos.FillSummaryHistoList = function() {
  if (WebLib.http_request.readyState == 4) {
    if (WebLib.http_request.status == 200) {
      try {
        var text = WebLib.http_request.responseText;
        var obj = document.getElementById("tree_list");
        if (obj != null) {
          obj.innerHTML = text;
          initTree();
          IMGC.loadingProgress('hide') ;
        }       
      }
      catch (err) {
//        alert ("Error detail: " + err.message); 
      }
    }
  }
}

//___________________________________________________________________________________
RequestHistos.FillModuleTree = function() {
  //alert ("FillModuleTree");
  if (WebLib.http_request.readyState == 4) {
    if (WebLib.http_request.status == 200) {
      try {
        var text = WebLib.http_request.responseText;
        var obj = document.getElementById("modtree_list");
        if (obj != null) {
          obj.innerHTML = text;
          initTree();
        }       
        IMGC.loadingProgress('hide') ;
      }
      catch (err) {
        //alert ("Error detail: " + err.message); 
      }
    }
  }
}

//___________________________________________________________________________________
RequestHistos.FillAlarmList = function() {
  if (WebLib.http_request.readyState == 4) {
    if (WebLib.http_request.status == 200) {
      try {
        var text = WebLib.http_request.responseText;
        var obj = document.getElementById("alarm_list");
        if (obj != null) {
          obj.innerHTML = text;
          initTree();
        }       
      }
      catch (err) {
        //alert ("Error detail: " + err.message); 
      }
    }
  }
}
