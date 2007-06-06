function RequestModuleHistoList() {
  var queryString;
  var url = getApplicationURL2();
  url += "/Request?";
  queryString = "RequestID=SingleModuleHistoList";
  url += queryString; 
  makeRequest(url, FillModuleHistoList);     
}
function RequestMEList(what) {
  var queryString;
  var url = getApplicationURL2();
  url += "/Request?";
  queryString = "RequestID=GetMEList";
  url += queryString; 
  if( what == 'CB' )
  { 
   makeRequest(url, FillMEList); 
  } else {
   makeRequest(url, ReturnMEList); 
  }    
}
function RequestSummaryHistoList() {
  var queryString;
  var url = getApplicationURL2();
  url += "/Request?";
  queryString = "RequestID=SummaryHistoList";
  var obj = document.getElementById("structure_name");
  var sname =  obj.options[obj.selectedIndex].value;
  queryString += '&StructureName='+sname;
  url += queryString; 
  makeRequest(url, FillSummaryHistoList);     
}
function RequestModuleTree() {
  var queryString;
  var url = getApplicationURL2();
  url += "/Request?";
  queryString = "RequestID=ModuleHistoList";
  var obj = document.getElementById("structure_for_module");
  var sname =  obj.options[obj.selectedIndex].value;
  queryString += '&StructureName='+sname;
  url += queryString; 
  makeRequest(url, FillModuleTree);     
}
function RequestAlarmList() {
  var queryString;
  var url = getApplicationURL2();
  url += "/Request?";
  queryString = "RequestID=AlarmList";
  var obj = document.getElementById("structure_for_alarm");
  var sname =  obj.options[obj.selectedIndex].value;
  queryString += '&StructureName='+sname;
  url += queryString; 
  makeRequest(url, FillAlarmList);     
}
function FillModuleHistoList() {
  if (http_request.readyState == 4) {
    if (http_request.status == 200) {
      try {


        var doc = http_request.responseXML;
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
function FillMEList() {
  if (http_request.readyState == 4) 
  {
    if (http_request.status == 200) 
    {
      try 
      {

        var doc = http_request.responseXML;
        var root = doc.documentElement;
        
        // Module Number select box
        // Histogram  select box
        var bobj = document.getElementById("monitoring_element_list");
        bobj.options.length = 0;

        var hrows = root.getElementsByTagName('Histo');
        for (var j = 0; j < hrows.length; j++) 
	{
          var name  = hrows[j].childNodes[0].nodeValue;
          var boption = new Option(name, name);
          try 
	  {
            bobj.add(boption, null);
          }
          catch (e) 
	  {
            bobj.add(boption, -1);
          }
        }
        enableButtons("UpdateTrackerMap") ;    
      }
      catch (err) {
        alert ("[RequestHistos.js::FillMEList()] Error detail: " + err.message); 
      }
    }
  }
}
function ReturnMEList() 
{
  if (http_request.readyState == 4) 
  {
    if (http_request.status == 200) 
    {
      try 
      {
        var doc   = http_request.responseXML;
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
function FillSummaryHistoList() {
  if (http_request.readyState == 4) {
    if (http_request.status == 200) {
      try {
        var text = http_request.responseText;
        var obj = document.getElementById("tree_list");
        if (obj != null) {
          obj.innerHTML = text;
          initTree();
        }       
      }
      catch (err) {
//        alert ("Error detail: " + err.message); 
      }
    }
  }
}
function FillModuleTree() {
  if (http_request.readyState == 4) {
    if (http_request.status == 200) {
      try {
        var text = http_request.responseText;
        var obj = document.getElementById("modtree_list");
        if (obj != null) {
          obj.innerHTML = text;
          initTree();
        }       
      }
      catch (err) {
//        alert ("Error detail: " + err.message); 
      }
    }
  }
}
function FillAlarmList() {
  if (http_request.readyState == 4) {
    if (http_request.status == 200) {
      try {
        var text = http_request.responseText;
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
