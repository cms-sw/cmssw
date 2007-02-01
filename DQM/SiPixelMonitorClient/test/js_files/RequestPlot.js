function DrawSelectedHistos() {
  //alert("BUTTON!");
  var queryString;
  var url = getApplicationURL2();
  url += "/Request?";
  queryString = 'RequestID=PlotAsModule';
  // Get Module Number
  var obj = document.getElementById("module_numbers");
  var value =  obj.options[obj.selectedIndex].value;
  queryString += '&ModId='+value;
  //alert("queryString="+queryString);
  var hist_opt = SetHistosAndPlotOption();
  if (hist_opt == " ") return;
  queryString += hist_opt;	
  // Get Canavs
  var canvas = document.getElementById("drawingcanvas");
  if (canvas == null) {
    alert("Canvas is not defined!");
    return;
  }
  queryString += '&width='+canvas.width+'&height='+canvas.height;
  url += queryString;
  //alert(" "+url);
  makeRequest(url, dummy);
  setTimeout('UpdatePlot()', 2000);   
}
//
//  -- Set Histograms and plotting options 
//    
function SetHistosAndPlotOption() {
   var dummy = " ";
   var qstring;
  // Histogram Names 
  var histos =GetSelectedHistos();
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
  //alert("histos:"+qstring);
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
  return qstring;
}  
function UpdatePlot() {
  var canvas = document.getElementById("drawingcanvas");

  var queryString = "RequestID=UpdatePlot";
  var url = getApplicationURL2();
  url = url + "/Request?";
  url = url + queryString;
  url = url + '&t=' + Math.random();
  canvas.src = url; 
}
function DrawSingleHisto(path){
  var url = getApplicationURL2();
  url += "/Request?";
  queryString = 'RequestID=PlotSingleHistogram';
  queryString += '&Path='+path;
  var canvas = document.getElementById("drawingcanvas");
  if (canvas == null) {
    alert("Canvas is not defined!");
    return;
  }
  queryString += '&width='+canvas.width+'&height='+canvas.height
  url += queryString;
  makeRequest(url, dummy);
   
  setTimeout('UpdatePlot()', 2000);     
}
function ReadStatus(path) {
  var url = getApplicationURL2();
  url += "/Request?";
  queryString = 'RequestID=ReadQTestStatus';
  queryString += '&Path='+path;
  url += queryString;
  makeRequest(url, FillStatus);
}
function FillStatus() {
  if (http_request.readyState == 4) {
    if (http_request.status == 200) {
      try {
        var text = http_request.responseText;
        var obj = document.getElementById("status_area");
        if (obj != null) {
          obj.innerHTML = text;
        }       
      }
      catch (err) {
//        alert ("Error detail: " + err.message); 
      }
    }
  }
}
