function RequestHistoList() {
  var queryString;
  var url = getApplicationURL2();
  url += "/Request?";
  if (document.getElementById("module_histos").checked) {
    queryString = "RequestID=SingleModuleHistoList";
    url += queryString; 
    makeRequest(url, FillModuleHistoList);     
  } else if (document.getElementById("track_histos").checked) {
    queryString = "RequestID=TrackHistoList";    
    url += queryString;
    makeRequest(url, FillTrackHistoList);     
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
//        alert(" rows = " + mrows.length);
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
//        alert(" rows = " + hrows.length);
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
function FillTrackHistoList() {
  if (http_request.readyState == 4) {
    if (http_request.status == 200) {
      try {


        var doc = http_request.responseXML;
        var root = doc.documentElement;
        
        // Histogram  select box
        var bobj = document.getElementById("histolistarea");
        bobj.options.length = 0;

        var hrows = root.getElementsByTagName('THisto');
//        alert(" rows = " + hrows.length);
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

/*function FillSummaryHistoList() {
  if (http_request.readyState == 4) {
    if (http_request.status == 200) {
      try {
        var doc = http_request.responseXML;
        var root = doc.documentElement;
        
        // Histogram  select box
        var obj = document.getElementById("histolistarea");
        obj.options.length = 0;

        var hrows = root.getElementsByTagName('SummaryHisto');
//        alert(" rows = " + hrows.length);
        for (var j = 0; j < hrows.length; j++) {
          var name  = hrows[j].childNodes[0].nodeValue;
          var option = new Option(name, name);
          try {
            obj.add(option, null);
          }
          catch (e) {
            obj.add(option, -1);
          }
        }
      }
      catch (err) {
        alert ("Error detail: " + err.message); 
      }
    }
  }
}*/
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
//        alert ("Error detail: " + err.message); 
      }
    }
  }
}
