var RequestHistos = {
    transport:null,	
    AJAX_REQUEST_STIMEOUT:2000,  // in milliseconds
    AJAX_REQUEST_LTIMEOUT:100000,
    AJAX_REQUEST_VTIMEOUT:500000, 

    CANVAS_TOTAL_WIDTH:630,
    CANVAS_TOTAL_HEIGHT:600,
    CANVAS_TOTAL_NUMBER:0,

    CANVAS_TEMP_WIDTH:0,
    CANVAS_TEMP_HEIGHT:0,
    CANVAS_TEMP_LEFT:0,
    CANVAS_TEMP_TOP:0,
    CANVAS_TEMP_ID:0

 
};
RequestHistos.SetSelectedValue = function() { 
   var aobj = document.getElementById('module_numbers');
   if (aobj != null) {
     var value =  aobj.options[aobj.selectedIndex].value;
     $('#module_number_edit').val(value);
  }
};
//
// - Error Response
//
RequestHistos.errorResponse = function (transport, status, errorThrown) {
    var message = 'Last ajax request failed, ' + 'status=' + status;
    if (status != 'timeout') message += "\nServer says:\n" + transport.responseText;
    alert(message);
};
//
// -- Disable 
//
RequestHistos.stopRKey = function(evt) {
  var evt  = (evt) ? evt : ((event) ? event : null);
  var node = (evt.target) ? evt.target : ((evt.srcElement) ? evt.srcElement :null);
  if ((evt.keyCode == 13) && (node.type=="text"))  {return false;}
};
//
// -- Make the Tabs visible/invisible
//
RequestHistos.ShowTabs = function(option) {
  $('#tabPanel1').attr('visibility',option);
};
//
// -- Make the buttons visible/invisible
//
RequestHistos.ShowButtons = function(option) {
  if (option) $('#open_tkmap').removeAttr("disabled");
  else $('#open_tkmap').attr("disabled","disabled");
};
//
// -- Show the progress bar with comment during waiting time
//
RequestHistos.SetProgressMessage = function(message) {
  var text_mess = '<B><Font size="+1" color="#8A2B31">Pl. wait, '+ message +'</Font></B>';
  $('#progress_message').html(text_mess);
};
//
//  -- Set Histograms and plotting options 
//    
RequestHistos.SetHistosAndPlotOption = function() {
   var dummy = " ";
   var qstring = " ";
  // Histogram Names 
  var hist_obj   = document.getElementById('histolistarea');
  var nhist = hist_obj.length;
  if (nhist == 0) {
    alert("Histogram List Area Empty!");
    return dummy;
  } else {
    for (var i = 0; i < nhist; i++) {
      if (hist_obj.options[i].selected) {
	if (qstring == " ") qstring  = '&histo='+ hist_obj.options[i].value;
        else        qstring += '&histo='+ hist_obj.options[i].value;
      }
    }
  }
  // Plot options for single histogram 
  if (nhist == 1) {
    // logy option
    if ($('logy').checked) {
      qstring += '&logy=true';
    }
  } 
  // Drawing option
  var obj1 = document.getElementById('drawing_options');
  var value1 =  obj1.options[obj1.selectedIndex].value;
  qstring += '&drawopt='+value1;
  return qstring;
} 
//
// -- Add Frames in the plotting Area
//
RequestHistos.AddFrames = function(nhist) {
    var wval;
    var hval;
    if (nhist == 1) {
        wval = 1;
        hval = 1;
    } else if (nhist == 2) {
        wval = 2;
        hval = 1;
    } else if (nhist > 2 && nhist <= 4) {
        wval = 2;
        hval = 2;
    } else if (nhist > 4 &&  nhist <= 6) {  
        wval = 2;
        hval = 3;
    } else if (nhist > 6 && nhist <= 9) {
        wval = 3;
        hval = 3;
    } else if (nhist > 9 && nhist <= 12) {
        wval = 4;
        hval = 3;
    } else if (nhist > 12 && nhist <= 16) {
        wval = 4;
        hval = 4;
    } else if (nhist > 16 && nhist <= 20) {
        wval = 5;
        hval = 4;
    } else if (nhist > 20 && nhist <= 25) {
        wval = 5;
        hval = 5;
    } else if (nhist > 25 && nhist <= 30) {
        wval = 6;
        hval = 5;
    } else if (nhist > 30) {
        wval = 6;
        hval = 6;
    }
    var height = RequestHistos.CANVAS_TOTAL_HEIGHT/hval;    
    var width  = RequestHistos.CANVAS_TOTAL_WIDTH/wval;
       
    $('#canvas').empty();
    var id = 0;
    for (var i = 1; i < wval+1; i++) { 
	for (var j = 1; j < hval+1; j++) { 
            id += 1;
            var left = (i-1) * width  + 12; 
            var top  = (j-1) * height + 20; 

	    var divid  = 'Div' + id;

            var parentdiv  = '#Div' + id;

	    var width1 = width - 15;
	    var height1 = height - 15;       
            
            var toppx = top+'px';
            var leftpx = left+'px';
	    $('<div/>').attr('id', divid)
		.css('float','left')
                .css('overflow','hidden')
		.css('background','#81787D')
		.appendTo('#canvas');

       	    var frameid = 'Frame' + id;	    
            $('<img/>').attr('id', frameid)
                .attr('class', 'small')
		.css('height', height)
		.css('width', width)
		.css('z-index', '0')
		.appendTo($(parentdiv));

	}
    }
};
//
// - Set Frame Properties
//
RequestHistos.SetFrameProperties = function(id,hpath) {
    var indx = hpath.lastIndexOf('/');
    var title = hpath.substring(indx+1,hpath.length) + '| Path : ' + hpath.substring(0,indx);
    var frameid = '#Frame' + id;                
    
    $(frameid).attr('title', title);

    $(frameid).mouseover(function(e){
       $(frameid).css('opacity','0.9');
    });
    $(frameid).mouseout(function(){
       $(frameid).css('opacity','1.0');
    });

    $(frameid).cluetip({splitTitle: '|', width: '400px'});

    $(frameid).click(function(){
       RequestHistos.ToggleFrame(id);
    });
};
//
// -- Toggle Invisible Canvas
//
RequestHistos.ToggleFrame = function(id) {

    var frameid = '#Frame'+id;
    var divid = '#Div'+id;

    if ($(frameid).attr('class') == 'small') {
      RequestHistos.CANVAS_TEMP_WIDTH  = $(frameid).css('width');
      RequestHistos.CANVAS_TEMP_HEIGHT = $(frameid).css('height');
      RequestHistos.CANVAS_TEMP_LEFT   = $(frameid).css('left');
      RequestHistos.CANVAS_TEMP_TOP    = $(frameid).css('top');
      RequestHistos.CANVAS_TEMP_ID     = id;

      $('div#canvas > div').hide();
      var top_val  =  $('#Div1').css('top');    
      var left_val =  $('#Div1').css('left');    
      $(frameid).css('top',top_val);
      $(frameid).css('left',left_val);
      $(frameid).css('z-index','10');
      $(frameid).css('width',RequestHistos.CANVAS_TOTAL_WIDTH);
      $(frameid).css('height',RequestHistos.CANVAS_TOTAL_HEIGHT);
      $(divid).show();
      $(frameid).attr('class', 'big');
    } else if ($(frameid).attr('class') == 'big') {
      $(frameid).css('top',RequestHistos.CANVAS_TEMP_TOP);
      $(frameid).css('left',RequestHistos.CANVAS_TEMP_LEFT);
      $(frameid).css('width',RequestHistos.CANVAS_TEMP_WIDTH);
      $(frameid).css('height',RequestHistos.CANVAS_TEMP_HEIGHT);
      $('div#canvas > div').show();
      $(frameid).attr('class', 'small');
    }
};
//
// -- Add Selected Summary Plots
//
RequestHistos.AddSelectedSummary = function(response, status) {
    try 
    {
      var  imageList = eval('('+response+')');

      var nhist = imageList.length;
      RequestHistos.AddFrames(nhist);
      for (var i = 1; i <= nhist; ++i) {  
        var url = WebLib.getApplicationURL();
        var plot_url  = url + "/" + imageList[i-1][0];
        var title = imageList[i-1][1];

        var frameid = '#Frame'+i;
	$(frameid).attr('src', plot_url); 
        RequestHistos.SetFrameProperties(i,title);
      } 
    }
    catch (err) {
      alert ("[RequestHistos.AddSelectedSummary] Error detail: " + err.message); 
    }
};
//
// -- Fill URLs of dufferent images in the canvas
//
RequestHistos.FillHistogramPath = function (response, status) {
    try {
        var root = response.documentElement;
        var hpath = root.getElementsByTagName('HPath');
        var hname = root.getElementsByTagName('HName');
        var path  = hpath[0].childNodes[0].nodeValue;	
        nhist = hname.length;
        if (hpath != 'NONE' && nhist > 0) {
	    // Create Image canvases
	    RequestHistos.AddFrames(nhist);
            RequestHistos.SetProgressMessage("drawing histogram(s)...");	
	    for (var i = 1; i < nhist+1; i++) {
                var name  = hname[i-1].childNodes[0].nodeValue;
                var full_path = path + '/' + name;
                var function_name = "RequestHistos.GetImage('"+i+"','"+full_path+"')";
                var frameid = '#Frame'+i;
                $(frameid).attr('src', 'images/blank.png');
                if (i < 3) {
                  setTimeout(function_name, 10000);
                } else {
                  setTimeout(function_name, 100);
                }
            }
            $('#progressbar').fadeOut(1000);    
	}
    }
    catch (err) {
        alert ("[RequestHistos.FillHistogramPath] Error detail: " + err.message); 
    }

};
//
// -- Get Image
//
RequestHistos.GetImage = function (id, path) {

  var queryString;	    
  queryString = "RequestID=GetImage";
  queryString += '&Path=' + path;
  queryString += '&width='+ RequestHistos.CANVAS_TOTAL_WIDTH;                         
  queryString += '&height='+ RequestHistos.CANVAS_TOTAL_HEIGHT;                        
  // date 
  var date = new Date();
  queryString += '&Time='+ date.getTime();
  var url = WebLib.getApplicationURLWithLID();
  url += queryString;
  var frameid = '#Frame'+id;
  $(frameid).attr('src', url); 
  RequestHistos.SetFrameProperties(id,path);
};
//
// -- Check whether the DQM Client is ready with histograms
//
RequestHistos.RequestReadyState = function(){
  var url         = WebLib.getApplicationURLWithLID();
  var queryString = "RequestID=IsReady";
  url             += queryString;
  RequestHistos.SetProgressMessage("requesting ready-state");

  RequestHistos.transport = $.ajax({
                url: url,
                type: 'GET',
                datatype: 'xml', 
                timeout: RequestHistos.AJAX_REQUEST_LTIMEOUT,
                success: RequestHistos.FillReadyState,
                error: RequestHistos.errorResponse
  });
};
//
// -- Fill Ready State
//
RequestHistos.FillReadyState = function(response, status) {
  try {
    var callAgain = false;
    var root      = response.documentElement;

    var aobj            = document.getElementById("summary_plot_type");
    aobj.options.length = 0;
    
    var hrows = root.getElementsByTagName('LName');
    var tkmap_types = root.getElementsByTagName('TKMapName');

    if (hrows.length < 1) {
      callAgain = true; 
    }  else {
      for (var i = 0; i < hrows.length; i++) {
        var l_name  = hrows[i].childNodes[0].nodeValue;
        var aoption = new Option(l_name, l_name);
        try {
          aobj.add(aoption, null);
//          var image_src = WebLib.getApplicationURL() + "/images/" + l_name + ".lis";
//          var title_src = WebLib.getApplicationURL() + "/images/" + l_name + "_titles.lis";
//          SlideShow.slideImageList[i] = image_src; 
//          SlideShow.slideTitleList[i] = title_src; 
        }
        catch (e) {
          aobj.add(aoption, -1);
        }
      }
//    SlideShow.nSlides = SlideShow.slideImageList.length;
      RequestHistos.FillTkMapOptions(response,status); 
      RequestHistos.ShowButtons(true);
      RequestHistos.ShowTabs('visible');
    }
    if (callAgain) setTimeout('RequestHistos.RequestReadyState()',20000);
  }
  catch (err) {
    alert (" [RequestHistos.FillReadyState] Error detail: " + err.message); 
  }
};
//
// -- Create Tracker Map
//
RequestHistos.OpenTrackerMapFrame = function() {
  var win = window.open('TrackerMapFrame.html');
  win.focus();            
};
//
// -- Create Tracker Structure options
//
RequestHistos.CreateTkStrustures = function(){
   var tk_struct = new Array("TIB/layer_1",
                             "TIB/layer_2", 
                             "TIB/layer_3",
                             "TIB/layer_4",
                             "TOB/layer_1", 
                             "TOB/layer_2",
                             "TOB/layer_3",
                             "TOB/layer_4", 
                             "TOB/layer_5",
                             "TOB/layer_6",
                             "TEC/MINUS/wheel_1",
                             "TEC/MINUS/wheel_2",
                             "TEC/MINUS/wheel_3",
                             "TEC/MINUS/wheel_4",
                             "TEC/MINUS/wheel_5",
                             "TEC/MINUS/wheel_6",
                             "TEC/MINUS/wheel_7",
                             "TEC/MINUS/wheel_8",
                             "TEC/MINUS/wheel_9",
                             "TEC/PLUS/wheel_1",
                             "TEC/PLUS/wheel_2",
                             "TEC/PLUS/wheel_3",
                             "TEC/PLUS/wheel_4",
                             "TEC/PLUS/wheel_5",
                             "TEC/PLUS/wheel_6",
                             "TEC/PLUS/wheel_7",
                             "TEC/PLUS/wheel_8",
                             "TEC/PLUS/wheel_9",
                             "TID/MINUS/wheel_1",
                             "TID/MINUS/wheel_2",
                             "TID/MINUS/wheel_3",
                             "TID/PLUS/wheel_1",
                             "TID/PLUS/wheel_2",
                             "TID/PLUS/wheel_3");



    var aobj            = document.getElementById('summ_struc_name');
    aobj.options.length = 0;
    var bobj            = document.getElementById('alarm_struc_name');
    bobj.options.length = 0;
    var cobj            = document.getElementById('mod_struc_name');
    cobj.options.length = 0;

    for (var i = 0; i < tk_struct.length; i++) {
      var opt_val = "MechanicalView/" + tk_struct[i];
      var option1 = new Option(tk_struct[i], opt_val); 
      var option2 = new Option(tk_struct[i], opt_val); 
      var option3 = new Option(tk_struct[i], opt_val); 
      aobj.add(option1, null);
      bobj.add(option2, null);
      cobj.add(option3, null);
    }
    aobj.selectedIndex = 0;
    bobj.selectedIndex = 0;
    cobj.selectedIndex = 0;
};
//
// -- Read Options for Tracker Map Creation
//
RequestHistos.ReadTkMapOptions = function() {
  var url          = WebLib.getApplicationURL();
  url              = url + "/sistrip_tkmap_option.xml"; 

  RequestHistos.transport = $.ajax({
                url: url,
                type: 'GET',
                datatype: 'xml', 
                timeout: RequestHistos.AJAX_REQUEST_LTIMEOUT,
                success: RequestHistos.FillTkMapOptions,
                error: RequestHistos.errorResponse
  });
};
//
// -- Fill Options for Tracker Map Creation
//
RequestHistos.FillTkMapOptions = function(response, status) {
    try 
    {
      var root  = response.documentElement;
      // TkMap Option Select Box
      var aobj  = document.getElementById("tkmap_option");
      aobj.options.length = 0;
      var mrows = root.getElementsByTagName('TkMapOption');
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
    }
    catch (err) {
      alert ("[RequestHistos.FillTkMapOptions] Error detail: " + err.message); 
    }
};
//
// -- Draw selected group plot from shifter page
//
RequestHistos.DrawSelectedSummary = function() {
  var tobj      = document.getElementById('summary_plot_type');
  var url = WebLib.getApplicationURL() ;
  var urlTitleList = url + '/images/' + tobj.value +'_titles.lis';
  var urlImageList = url + '/images/' + tobj.value +'.lis';
  RequestHistos.SetProgressMessage("drawing selected group of plots");  
  RequestHistos.transport = $.ajax({
                url: urlImageList,
                type: 'GET',
                datatype: 'text', 
                timeout: RequestHistos.AJAX_REQUEST_LTIMEOUT,
                success: RequestHistos.AddSelectedSummary,
                error: RequestHistos.errorResponse
  });
};
//
// Check Quality Test Results (Lite)
//
RequestHistos.CheckQualityTestResultsLite = function() {
  var queryString  = "RequestID=CheckQTResults";
  queryString     += '&InfoType=Lite';
  var url          = WebLib.getApplicationURLWithLID();
  url              = url + queryString; 
  RequestHistos.SetProgressMessage("requesting QTest summary");
  RequestHistos.transport = $.ajax({
                url: url,
                type: 'GET',
                datatype: 'html', 
                timeout: RequestHistos.AJAX_REQUEST_LTIMEOUT,
                success: RequestHistos.FillTextStatus,
                error: RequestHistos.errorResponse
  });
};
//
// Check Quality Test Results (Expert)
//
RequestHistos.CheckQualityTestResultsDetail = function() {
  var queryString  = "RequestID=CheckQTResults";
  queryString     += '&InfoType=Detail';
  var url          = WebLib.getApplicationURLWithLID();
  url              = url + queryString; 
  RequestHistos.SetProgressMessage("requesting QTest summary");
  RequestHistos.transport = $.ajax({
                url: url,
                type: 'GET',
                datatype: 'html', 
                timeout: RequestHistos.AJAX_REQUEST_LTIMEOUT,
                success: RequestHistos.FillTextStatus,
                error: RequestHistos.errorResponse
  });
};
//
// -- Fill the status of QTest in the status list area
//
RequestHistos.FillTextStatus = function(response, status) 
{
  try {
      $('#summary_status_area').html(response);
    }
    catch (err) {
      alert ("[RequestHistos.FillTextStatus] Error detail: " + err.message); 
    }
};
//
// -- Request Readout/Control Tree
//
RequestHistos.RequestNonGeomeHistoList = function()
{
  var queryString;
  var url      = WebLib.getApplicationURLWithLID();
  queryString  = "RequestID=NonGeomHistoList";
  var obj      = document.getElementById("comm_type_tag");
  var fname    = obj.options[obj.selectedIndex].value;
  queryString += '&FolderName='+fname;
  url         += queryString; 
  RequestHistos.SetProgressMessage("requesting non-geometric histogram tree");
  RequestHistos.transport = $.ajax({
                url: url,
                type: 'GET',
                datatype: 'text',
                timeout: RequestHistos.AJAX_REQUEST_LTIMEOUT,
                success: RequestHistos.FillNonGeomHistoList,
                error: RequestHistos.errorResponse
  });
};
//
// -- Fill the readout/control tree in the list area
//
RequestHistos.FillNonGeomHistoList = function(response, status) {
    try {
        $('#non_geo_hlist').html(response);
        $('#non_geo_tree').treeview({animated: "fast",
                                     collapsed: true,
                                     unique: true});
    }       
    catch (err) {
      alert ("[RequestHistos.FillNonGeometricHistoList] Error detail: " + err.message); 
    }
};
//
// Request Summary Histogram List
//
RequestHistos.RequestSummaryHistoList = function(){
  var queryString;
  var url      = WebLib.getApplicationURLWithLID();
  queryString  = "RequestID=SummaryHistoList";
  var obj      = document.getElementById("summ_struc_name");
  var sname    = obj.options[obj.selectedIndex].value;
  queryString += '&StructureName='+sname;
  url         += queryString; 
  RequestHistos.SetProgressMessage("requesting summary histogram tree");
  RequestHistos.transport = $.ajax({
                url: url,
                type: 'GET',
                datatype: 'text',
                timeout: RequestHistos.AJAX_REQUEST_LTIMEOUT,
                success: RequestHistos.FillSummaryHistoList,
                error: RequestHistos.errorResponse
  });
};
//
// -- Fill the Summary tree in the list area
//
RequestHistos.FillSummaryHistoList = function(response, status) {
    try {
        $('#summary_hlist').html(response);
        $('#summary_histo_tree').treeview({animated: "fast",
                                     collapsed: true,
                                     unique: true});
    }       
    catch (err) {
      alert ("[RequestHistos.FillSummaryHistoList] Error detail: " + err.message); 
    }
};
//
// -- Draw CondDB Histos for Module
//
RequestHistos.DrawSummaryHistogram = function(path){
  var queryString;
  var url = WebLib.getApplicationURLWithLID();
  queryString = "RequestID=PlotHistogramFromPath";
  queryString += '&Path='+path;
  queryString += '&width='+RequestHistos.CANVAS_TOTAL_WIDTH+
                 '&height='+RequestHistos.CANVAS_TOTAL_HEIGHT;
  url += queryString;
  RequestHistos.SetProgressMessage("requesting summary plots");
  RequestHistos.transport = $.ajax({
          url: url,
          type: 'GET',
          datatype: 'xml',
          timeout: RequestHistos.AJAX_REQUEST_VTIMEOUT,
          success: RequestHistos.FillHistogramPath,
          error: RequestHistos.errorResponse
  });
};
//
// -- Draw CondDB Histos for Module
//
RequestHistos.DrawLayerCondDBHisto = function()
{
  var queryString;
  var url = WebLib.getApplicationURLWithLID();
  queryString = "RequestID=PlotLayerCondDBHistos";
  var obj      = document.getElementById("summ_struc_name");
  var sname    = obj.options[obj.selectedIndex].value;
  queryString += '&StructureName='+sname;
  var option   = 'NoiseFromCondDB,FractionOfBadStripsFromCondDB';
  queryString += '&option='+option;  
  queryString += '&width='+RequestHistos.CANVAS_TOTAL_WIDTH+
                 '&height='+RequestHistos.CANVAS_TOTAL_HEIGHT;
  url += queryString;
  RequestHistos.SetProgressMessage("requesting CondDB plot(s)");
  RequestHistos.transport = $.ajax({
          url: url,
          type: 'GET',
          datatype: 'xml',
          timeout: RequestHistos.AJAX_REQUEST_VTIMEOUT,
          success: RequestHistos.FillHistogramPath,
          error: RequestHistos.errorResponse
  });
};
//
// Request Summary Histogram List
//
RequestHistos.RequestAlarmList = function(){
  var queryString;
  var url      = WebLib.getApplicationURLWithLID();
  queryString  = "RequestID=AlarmList";
  var obj      = document.getElementById("alarm_struc_name");
  var sname    = obj.options[obj.selectedIndex].value;
  queryString += '&StructureName='+sname;
  url         += queryString; 
  RequestHistos.SetProgressMessage("requesting alarm tree");
  RequestHistos.transport = $.ajax({
                url: url,
                type: 'GET',
                datatype: 'text',
                timeout: RequestHistos.AJAX_REQUEST_LTIMEOUT,
                success: RequestHistos.FillAlarmList,
                error: RequestHistos.errorResponse
  });
};
//
// -- Fill the Summary tree in the list area
//
RequestHistos.FillAlarmList = function(response, status) {
    try {
        $('#alarm_list').html(response);
        $('#alarm_tree').treeview({animated: "fast",
                                   collapsed: true,
                                   unique: true});
    }       
    catch (err) {
      alert ("[RequestHistos.FillAlarmList] Error detail: " + err.message); 
    }
};
//
// -- Read status message from QTest
//
RequestHistos.ReadAlarmStatus = function(path) {
  var url      = WebLib.getApplicationURLWithLID();
  queryString  = 'RequestID=ReadQTestStatus';
  queryString += '&Path='+path;
  queryString += '&width='+RequestHistos.CANVAS_TOTAL_WIDTH+
                 '&height='+RequestHistos.CANVAS_TOTAL_HEIGHT;
  url         += queryString;
  RequestHistos.SetProgressMessage("reading alarm status");
  RequestHistos.transport = $.ajax({
                url: url,
                type: 'GET',
                datatype: 'xml',
                timeout: RequestHistos.AJAX_REQUEST_LTIMEOUT,
                success: RequestHistos.FillAlarmStatus,
                error: RequestHistos.errorResponse
  });
};
//
// -- Fill status message from QTest in the status list area
//
RequestHistos.FillAlarmStatus = function(response, status){
   try {
      var root = response.documentElement;
      var mrows = root.getElementsByTagName('Status');
      if (mrows.length > 0) {
        var stat  = mrows[0].childNodes[0].nodeValue;
        $('#alarm_status_area').html(stat);
      }
      hpath = root.getElementsByTagName('HPath');
      hname = root.getElementsByTagName('HName');
      nhist = hname.length; 
      if (nhist > 1) {
        // Create Image canvases
        RequestHistos.AddFrames(nhist);
        RequestHistos.SetProgressMessage("drawing histogram(s)...");			
        for (var i = 1; i < nhist+1; i++) {
          // path
          var path  = hpath[0].childNodes[0].nodeValue;
          var name  = hname[i-1].childNodes[0].nodeValue;
          var full_path = path + '/' + name;
          var function_name = "RequestHistos.GetImage('"+i+"','"+full_path+"')";
          var frameid = '#Frame'+i;
          $(frameid).attr('src', 'images/blank.png');
          if (path != "NONE") 
            if (i < 3) {
                (setTimeout(function_name, 10000));
            } else {
                (setTimeout(function_name, 100));
          }
        }
      }
      $('#progressbar').fadeOut(1000);
   }
   catch (err) {
     alert ("[RequestHistos.FillAlarmStatus] Error detail: " + err.message); 
   }
};
//
// -- Get list of histogram names according to the option selected
//
RequestHistos.RequestHistoList = function() {
  var queryString;
  var url = WebLib.getApplicationURLWithLID();
  if ($('#module_histos').attr('checked')) {
    queryString = "RequestID=SingleModuleHistoList";
    var obj = document.getElementById('mod_struc_name');
    var sname    = obj.options[obj.selectedIndex].value;
    queryString += '&FolderName='+sname;    
    url        += queryString; 
    RequestHistos.SetProgressMessage("requesting module & histogram list...");	
    RequestHistos.transport = $.ajax({
                 url: url,
                 type: 'GET',
                 datatype: 'xml',
                 timeout: RequestHistos.AJAX_REQUEST_LTIMEOUT,
                 success: RequestHistos.FillModuleHistoList,
                 error: RequestHistos.errorResponse	
    });
  } else if ($('#global_histos').attr('checked')) {
    queryString = "RequestID=GlobalHistoList";    
    var value =  $('#ghisto_path').val();
    queryString += '&GlobalFolder='+value;
    url        += queryString;
    RequestHistos.SetProgressMessage("requesting global histogram list...");	
    RequestHistos.transport = $.ajax({
                 url: url,
                 type: 'GET',
                 datatype: 'xml',
                 timeout: RequestHistos.AJAX_REQUEST_LTIMEOUT,
                 success: RequestHistos.FillGlobalHistoList,
                 error: RequestHistos.errorResponse	
    });

  }
};
//
// -- Fill list of modules and histogram names in th list area
//
RequestHistos.FillModuleHistoList = function(response, status) {
    try 
    {
      var root  = response.documentElement;
        
      // Module Number select box
      var aobj  = document.getElementById('module_numbers');

      aobj.options.length = 0;
        
      var mrows = root.getElementsByTagName('ModuleNum');
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
      var cobj = document.getElementById('module_number_edit');
      if (cobj != null) {
        cobj.value = aobj.options[0].value;;
      }   
      // Histogram  select box
      var bobj = document.getElementById('histolistarea');
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
      alert ("[RequestHistos.FillModuleHistoList] Error detail: " + err.message); 
    }
}
//
// -- Fill names of the global histogram
//
RequestHistos.FillGlobalHistoList = function(response, status) {
    try 
    {
      var root = response.documentElement;
       
      // Histogram  select box
      var bobj = document.getElementById('histolistarea');
      bobj.options.length = 0;

      var hrows = root.getElementsByTagName('GHisto');
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
};
RequestHistos.DrawSelectedHistos = function() {
  var queryString;
  var url = WebLib.getApplicationURLWithLID();
  if ($('#module_histos').attr('checked')) {
    queryString = "RequestID=PlotAsModule";
    // Get Module Number
    var obj      = document.getElementById('module_number_edit');
    var value    = obj.value;
    queryString += '&ModId='+value;
  } else if ($('#global_histos').attr('checked')) {
    queryString  = "RequestID=PlotGlobalHisto";
    var obj = document.getElementById('ghisto_path');
    var value =  obj.value;
    queryString += '&GlobalFolder='+value;
  }
  var hist_opt   = RequestHistos.SetHistosAndPlotOption();
  if (hist_opt == " ") return;
  queryString   += hist_opt;
  queryString += '&width='+RequestHistos.CANVAS_TOTAL_WIDTH+
                 '&height='+RequestHistos.CANVAS_TOTAL_HEIGHT;
  url += queryString;
  RequestHistos.SetProgressMessage("requesting plots");		
  RequestHistos.transport = $.ajax({
          url: url,
          type: 'GET',
          datatype: 'xml',
          timeout: RequestHistos.AJAX_REQUEST_VTIMEOUT,
          success: RequestHistos.FillHistogramPath,
          error: RequestHistos.errorResponse
  });
};
//
// -- Draw CondDB Histos for Module
//
RequestHistos.DrawModuleCondDBHisto = function() {
  if ($('#global_histos').attr('checked')) {
    alert("Global Plot option is selected!! Select Modules");
    return;
  }
  var queryString;
  var url = WebLib.getApplicationURLWithLID();
  queryString = "RequestID=PlotModuleCondDBHistos";
  // Get Module Number
  var obj      = document.getElementById('module_number_edit');
  var value    = obj.value;
  queryString += '&ModId='+value;
  var option = 'NoiseFromCondDB';
  queryString += '&option='+option;  
  queryString += '&width='+RequestHistos.CANVAS_TOTAL_WIDTH+
                 '&height='+RequestHistos.CANVAS_TOTAL_HEIGHT;

  url += queryString;
  RequestHistos.SetProgressMessage("requesting CondDB plots");	
  RequestHistos.transport = $.ajax({
          url: url,
          type: 'GET',
          datatype: 'xml',
          timeout: RequestHistos.AJAX_REQUEST_VTIMEOUT,
          success: RequestHistos.FillHistogramPath,
          error: RequestHistos.errorResponse
  });
};
//
// Update TrackerMap Option
//
RequestHistos.UpdateTrackerMapOption = function() {

  var queryString;
  var url      = WebLib.getApplicationURLWithLID();
  queryString  = "RequestID=UpdateTrackerMapOption";
  var obj = document.getElementById('tkmap_option');  
  var sname    = obj.options[obj.selectedIndex].value;
  queryString += '&Option=' + sname;
  url += queryString;
  RequestHistos.transport = $.ajax({
          url: url,
          type: 'GET',
          datatype: 'xml',
          timeout: RequestHistos.AJAX_REQUEST_VTIMEOUT,
          success: RequestHistos.DummyAction,
          error: RequestHistos.errorResponse
  });
};
//
// -- Open Tracker Map
//
RequestHistos.OpenTrackerMap = function()
{
  var win = window.open('dqmtmapviewer.html');
  win.focus();
}
//
// -- TkMap
//
RequestHistos.RequestTkMapHistos = function(det_id) {
  var queryString;
  var url = WebLib.getApplicationURLWithLID();
  queryString = "RequestID=PlotTkMapHistogram";
  queryString += '&ModId='+det_id;
  queryString += '&width='+RequestHistos.CANVAS_TOTAL_WIDTH+
                 '&height='+RequestHistos.CANVAS_TOTAL_HEIGHT;
  url += queryString;
  RequestHistos.SetProgressMessage("requesting plots");		
  RequestHistos.transport = $.ajax({
          url: url,
          type: 'GET',
          datatype: 'xml',
          timeout: RequestHistos.AJAX_REQUEST_VTIMEOUT,
          success: RequestHistos.FillHistogramPath,
          error: RequestHistos.errorResponse
  });
}
//
// Dummy Function
//
RequestHistos.Dummy = function() {
};
//
// Main
//
$(document).ready(function(){
  $('#tabPanel1').tabs();
  $().ajaxStart(function() { $('#progressbar').fadeIn(1000); });
  $().ajaxStop(function() { $('#progressbar').fadeOut(1000);  });

  $('#canvas').height(RequestHistos.CANVAS_TOTAL_HEIGHT*1.05);
  $('#canvas').width(RequestHistos.CANVAS_TOTAL_WIDTH);

  document.onkeypress = RequestHistos.stopRKey;
  RequestHistos.CreateTkStrustures();
//  RequestHistos.ReadTkMapOptions();
  RequestHistos.ShowButtons(false);
  RequestHistos.ShowTabs('hidden');  
  RequestHistos.RequestReadyState();
});
