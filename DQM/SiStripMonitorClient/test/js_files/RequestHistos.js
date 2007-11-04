var RequestHistos = {};
//
// -- Get list of histogram names according to the option selected
//
RequestHistos.RequestHistoList = function() 
{
  var queryString;
  var url = WebLib.getApplicationURL2();
  if (document.getElementById("module_histos").checked) {
    queryString = "RequestID=SingleModuleHistoList";
    url        += queryString; 
    var retVal = new Ajax.Request(url,                    
 	 	                 {			  
 	 		          method: 'get',	  
 			          parameters: '', 
 			          onSuccess: RequestHistos.FillModuleHistoList
 			         });
//    WebLib.makeRequest(url, RequestHistos.FillModuleHistoList);
    CommonActions.ShowProgress("visible", "Module Histogram List");     
  } else if (document.getElementById("global_histos").checked) {
    queryString = "RequestID=GlobalHistoList";    
    url        += queryString;
    var retVal = new Ajax.Request(url,                    
 	 	                 {			  
 	 		          method: 'get',	  
 			          parameters: '', 
 			          onSuccess: RequestHistos.FillGlobalHistoList
 			         });
//    WebLib.makeRequest(url, RequestHistos.FillGlobalHistoList);     
    CommonActions.ShowProgress("visible", "Global Histogram List");
  }
}
//
// -- Request summary histogram tree
//
RequestHistos.RequestSummaryHistoList = function()
{
  var queryString;
  var url      = WebLib.getApplicationURL2();
  queryString  = "RequestID=SummaryHistoList";
  var obj      = document.getElementById("structure_name");
  var sname    = obj.options[obj.selectedIndex].value;
  queryString += '&StructureName='+sname;
  url         += queryString; 
  var retVal = new Ajax.Request(url,
                               {           
                  		method: 'get',	  
 			        parameters: '', 
 			        onSuccess: RequestHistos.FillSummaryHistoList
 			       });
//  WebLib.makeRequest(url, RequestHistos.FillSummaryHistoList);     
  CommonActions.ShowProgress("visible", "Summary Histogram Tree");
}
//
// -- Request the alarm tree
//
RequestHistos.RequestAlarmList = function()
{
  var queryString;
  var url      = WebLib.getApplicationURL2();
  queryString  = "RequestID=AlarmList";
  var obj      = document.getElementById("structure_for_alarm");
  var sname    = obj.options[obj.selectedIndex].value;
  queryString += '&StructureName='+sname;
  url         += queryString; 
  var retVal = new Ajax.Request(url,
                               {           
                  		method: 'get',	  
 			        parameters: '', 
 			        onSuccess: RequestHistos.FillAlarmList
 			       });

//  WebLib.makeRequest(url, RequestHistos.FillAlarmList);     
  CommonActions.ShowProgress("visible", "Alarm Tree");
}
//
// -- Fill list of modules and histogram names in th list area
//
RequestHistos.FillModuleHistoList = function(transport) 
{
    CommonActions.ShowProgress("hidden");
    try 
    {
      var doc   = transport.responseXML;
      var root  = doc.documentElement;
        
      // Module Number select box
      var aobj  = document.getElementById("module_numbers");

      aobj.options.length = 0;
        
       var mrows = root.getElementsByTagName('ModuleNum');
//        alert(" rows = " + mrows.length);
      for (var i = 0; i < mrows.length; i++) {
        var mnum = mrows[i].childNodes[0].nodeValue;
        var aoption = new Option(mnum, mnum);
        try 
        {
          aobj.add(aoption, null);
        }
        catch (e) {
          aobj.add(aoption, -1);
        }
      }
      // Select the first option and set to editable text  
      var cobj = document.getElementById("module_number_edit");
      if (cobj != null) {
        cobj.value = aobj.options[0].value;;
      }   
      // Histogram  select box
      var bobj = document.getElementById("histolistarea");
      bobj.options.length = 0;

      var hrows = root.getElementsByTagName('Histo');
      // alert(" rows = " + hrows.length);
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
      alert ("[RequestHistos.FillModuleHistoList] Error detail: " + err.message); 
    }
}
//
// -- Fill names of the global histogram
//
RequestHistos.FillGlobalHistoList = function(transport) 
{
    CommonActions.ShowProgress("hidden");
    try 
    {
      var doc  = transport.responseXML;
      var root = doc.documentElement;
       
      // Histogram  select box
      var bobj = document.getElementById("histolistarea");
      bobj.options.length = 0;

      var hrows = root.getElementsByTagName('GHisto');
      // alert(" rows = " + hrows.length);
      for (var j = 0; j < hrows.length; j++) {
        var name    = hrows[j].childNodes[0].nodeValue;
        var boption = new Option(name, name);
        try 
        {
          bobj.add(boption, null);
        }
        catch (e) {
          bobj.add(boption, -1);
        }
      }
    }
    catch (err) {
      alert ("[RequestHistos.FillGlobalHistoList] Error detail: " + err.message); 
    }
}
//
// -- Fill the summary tree in the list area
//
RequestHistos.FillSummaryHistoList = function(transport) 
{
    CommonActions.ShowProgress("hidden");
    try {
      var text = transport.responseText;
      var obj  = document.getElementById("tree_list");
      if (obj != null) {
        obj.innerHTML = text;
        initTree();
      }       
    }
    catch (err) {
    // alert ("[RequestHistos.FillSummaryHistoList] Error detail: " + err.message); 
    }
}
//
// -- Fill alarm tree in the list area
//
RequestHistos.FillAlarmList = function(transport) 
{
    CommonActions.ShowProgress("hidden");
    try {
      var text = transport.responseText;
      var obj = document.getElementById("alarm_list");
      if (obj != null) {
        obj.innerHTML = text;
        initTree();
      }       
    }
    catch (err) {
    // alert ("[RequestHistos.FillAlarmList] Error detail: " + err.message); 
    }
}
//
// -- Draw selected histograms from the list area
//
RequestHistos.DrawSelectedHistos = function() 
{
  var queryString;
  var url = WebLib.getApplicationURL2();
  if (document.getElementById("module_histos").checked) {
    queryString = "RequestID=PlotAsModule";
    // Get Module Number
    var obj      = document.getElementById("module_number_edit");
    var value    = obj.value;
    queryString += '&ModId='+value;
  } else if (document.getElementById("global_histos").checked) {
    queryString  = "RequestID=PlotGlobalHisto";    
  }
  var hist_opt   = RequestHistos.SetHistosAndPlotOption();
  if (hist_opt == " ") return;
  queryString   += hist_opt;	
  // Get Canavs
  var canvas     = document.getElementById("drawingcanvas");
  if (canvas == null) {
    alert("Canvas is not defined!");
    return;
  }
  queryString += '&width='+canvas.width+'&height='+canvas.height;
  url += queryString;
  var retVal = new Ajax.Request(url,
                               {           
                  		method: 'get',	  
 			        parameters: '', 
 			        onComplete: ''
 			       });

//  WebLib.makeRequest(url, null);
  CommonActions.ShowProgress('visible', 'Selected Plot');
  setTimeout('RequestHistos.UpdatePlot()', 2000);   
}
//
//  -- Set Histograms and plotting options 
//    
RequestHistos.SetHistosAndPlotOption = function() {
   var dummy = " ";
   var qstring;
  // Histogram Names 
  var histos = CommonActions.GetSelectedHistos();
  if (histos.length == 0) {
    alert("Plot(s) not defined!");
    return dummy;
  }
  //  
  var nhist = histos.length;
  // alert(" "+nhist);
  for (var i = 0; i < nhist; i++) {
    if (i == 0) qstring = '&histo='+histos[i];
    else qstring += '&histo='+histos[i];
  }

  // Rows and columns
  var nr = 1;
  var nc = 1;
  if (nhist == 1) {
    // logy option
    if (document.getElementById("logy").checked) {
      qstring += '&logy=true';
    }
    obj = document.getElementById("x-low");
    value = parseFloat(obj.value);
    if (!isNaN(value)) qstring += '&xmin=' + value;

    obj = document.getElementById("x-high");
    value = parseFloat(obj.value);
    if (!isNaN(value)) qstring += '&xmax=' + value;
  } else {
    if (document.getElementById("multizone").checked) {
      obj = document.getElementById("nrow");
      nr =  parseInt(obj.value);
      if (isNaN(nr)) {
        nr = 1;
      }
      obj = document.getElementById("ncol");
      nc = parseInt(obj.value);
      if (isNaN(nc)) {
        nc = 2;       
      }
    }
    if (nr*nc < nhist) {
      if (nhist <= 10) {
        nc = 2;
      } else if (nhist <= 20) {
        nc = 3;
      } else if (nhist <= 30) {
         nc = 4;
      } 		
       nr = Math.ceil(nhist*1.0/nc);
    }
    qstring += '&cols=' + nc + '&rows=' + nr;       
  }
  // Drawing option
  var obj1 = document.getElementById("drawing_options");
  var value1 =  obj1.options[obj1.selectedIndex].value;
  qstring += '&drawopt='+value1;
  return qstring;
} 
//
// -- Get last plot from the server
// 
RequestHistos.UpdatePlot = function()
{
  var canvas = document.getElementById("drawingcanvas");

  var queryString = "RequestID=UpdatePlot";
  var url = WebLib.getApplicationURL2();
  url = url + queryString;
  url = url + '&t=' + Math.random();
  canvas.src = url; 
  CommonActions.ShowProgress('hidden');
}
//
// -- Draw single histogram from path
// 
RequestHistos.DrawSingleHisto = function(path)
{
  var url      = WebLib.getApplicationURL2();
  queryString  = 'RequestID=PlotHistogramFromPath';
  queryString += '&Path='+path;
  var canvas   = document.getElementById("drawingcanvas");
  if (canvas == null) {
    alert("Canvas is not defined!");
    return;
  }
  queryString += '&width='+canvas.width+'&height='+canvas.height;
  queryString += '&histotype=summary';
  url         += queryString;
  var retVal = new Ajax.Request(url,
                               {           
                  		method: 'get',	  
 			        parameters: '', 
 			        onComplete: ''
 			       });

//  WebLib.makeRequest(url, null);
  CommonActions.ShowProgress('visible', 'Selected Plot');   
  setTimeout('RequestHistos.UpdatePlot()', 2000);     
}
//
// -- Read status message from QTest
//
RequestHistos.ReadStatus = function(path) 
{
  var url      = WebLib.getApplicationURL2();
  queryString  = 'RequestID=ReadQTestStatus';
  queryString += '&Path='+path;
  url         += queryString;
  CommonActions.ShowProgress('visible', 'Status Message');
  var retVal = new Ajax.Request(url,
                               {           
                  		method: 'get',	  
 			        parameters: '', 
 			        onSuccess: RequestHistos.FillStatus
 			       });

//  WebLib.makeRequest(url, RequestHistos.FillStatus);
}
//
// -- Fill status message from QTest in the status list area
//
RequestHistos.FillStatus = function(transport) {
   CommonActions.ShowProgress('hidden');
   try {
      var doc = transport.responseXML;
      var root = doc.documentElement;
      var mrows = root.getElementsByTagName('Status');
      if (mrows.length > 0) {
        var stat  = mrows[0].childNodes[0].nodeValue;
        var obj = document.getElementById("status_area");
        if (obj != null) {
          obj.innerHTML = stat;
        }       
      }
      mrows = root.getElementsByTagName('HPath');
      if (mrows.length > 0) {
        var hpath  = mrows[0].childNodes[0].nodeValue;
        if (hpath != "NONE") RequestHistos.DrawQTestHisto(hpath);
      }
    }
    catch (err) {
//      alert ("Error detail: " + err.message); 
    }
}
//
// -- Draw Histogram used for QTest
//
RequestHistos.DrawQTestHisto = function(path)
{
  var url      = WebLib.getApplicationURL2();
  queryString  = 'RequestID=PlotHistogramFromPath';
  queryString += '&Path='+path;
  var canvas   = document.getElementById("drawingcanvas");
  if (canvas == null) {
    alert("Canvas is not defined!");
    return;
  }
  queryString  += '&width='+canvas.width+'&height='+canvas.height;
  queryString  += '&histotype=qtest';
  url          += queryString;
  var retVal = new Ajax.Request(url,
                               {           
                  		method: 'get',	  
 			        parameters: '', 
 			        onComplete: ''
 			       });

//  WebLib.makeRequest(url, null);
  CommonActions.ShowProgress('visible', 'Selected Plot');
   
  setTimeout('RequestHistos.UpdatePlot()', 2000);     
}
//
// -- Draw selected group plot from shifter page
//
RequestHistos.DrawSelectedSummary = function() 
{
  var canvas    = document.getElementById("drawingcanvas");
  if (canvas == null) {
    alert("Canvas is not defined!");
    return;
  } 
  var tobj      = document.getElementById("summary_plot_type");
  var image_src = SlideShow.slideList[tobj.selectedIndex];
  image_src    += '?t=' + Math.random();  //Should start with "?"
  canvas.src    = image_src;   
} 
//
// Check Quality Test Results (Lite)
//
RequestHistos.CheckQualityTestResultsLite = function() 
{
  var queryString  = "RequestID=CheckQTResults";
  queryString     += '&InfoType=Lite';
  var url          = WebLib.getApplicationURL2();
  url              = url + queryString; 
  var retVal = new Ajax.Request(url,
                               {           
                  		method: 'get',	  
 			        parameters: '', 
 			        onSuccess: RequestHistos.FillTextStatus
 			       });
  
//  WebLib.makeRequest(url, RequestHistos.FillTextStatus); 
}
//
// Check Quality Test Results (Expert)
//
RequestHistos.CheckQualityTestResultsDetail = function() {
  var queryString  = "RequestID=CheckQTResults";
  queryString     += '&InfoType=Detail';
  var url          = WebLib.getApplicationURL2();
  url              = url + queryString; 
  var retVal = new Ajax.Request(url,
                               {           
                  		method: 'get',	  
 			        parameters: '', 
 			        onSuccess: RequestHistos.FillTextStatus
 			       });

//  WebLib.makeRequest(url, RequestHistos.FillTextStatus); 
}
//
// -- Fill the status of QTest in the status list area
//
RequestHistos.FillTextStatus = function(transport) 
{
  try {
    var text = transport.responseText;
      CommonActions.FillText("summary_status_area", text);
    }
    catch (err) {
//      alert ("Error detail: " + err.message); 
    }
}
