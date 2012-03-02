//
//var baseURL="http://localhost/khotilov/plots/";
//var baseURL="http://hepcms1.physics.tamu.edu/khotilov/plots/";
var baseURL="http://hepr8.physics.tamu.edu/vadim/cms/mual/plots/";
//var ResultsFolder="";
var ResultsFolder = baseURL + "commissioning2/";
//var ResultsFolder = baseURL + "hw/";
var refURL=baseURL+"ref.plots";

var RunsList="runs_list.js";
var TestsList="canvases_list.js";
var MUList="mu_list.js";
//var CSCMap="csc_map_new.js";
//var DDUMap="ddu_csc_map_new.js";
//var VMEMap="vme_pc_map.js";
var TestsMap="tests_map.js";
//var ResFolders="results_folders.js";
var CSCCounters="csc_counters.js";
var DQMReport="No Summary Report";
var TextReport="dqm_report.txt";
var JSONReport="dqm_report.js";
var reportWindow = "";

var imgFormat=".png";
var RunNumber="";
var Folder="";
var RunInfoIdx = -1;
var Canvas="";
var Scope="";
var isFolderValid=false;
var postfix = ".plots";

var minImageHeight=100;
var maxImageHeight=1200;
var imgHeight=600;
var imgWidth=800;

var RUNS=false;	// list of Runs
//var TREE_RUNS=false; // actually list of Tests/Canvases
//var CSCMAP=false;
var MU_LIST=false;
//var DDUMAP=false;
//var DDUMAPnew=false;
//var VMEMAP=false;
var RESULTS_FOLDERS=false;
var CSC_COUNTERS=false;
var TESTS_MAP=false;

var Log=new Array();
var logLevel="all";

var isIE = false;

//var fNewDDUMap = true;

var selectedObjectList=new Array();

var testsShowLists = new Array();
testsShowLists["Custom"] = new Array();
var runsShowLists = new Array();
runsShowLists["Custom"] = new Array();
var foldersShowLists = new Array();
foldersShowLists["Selected Only"] = new Array();
foldersShowLists["All CSCs"] = new Array();
// foldersShowLists["Custom"] = new Array();
var folderSListIdx=0;
var fOpenExternal=false;
var fShowReference;
var fShowIter1;
var loopFoldersFirst = true;

var EMUs = new Array();
var DDUs = new Array();
var CSCs = new Array();

var SystemViewTabs = new Array();

var num_chambers=0;
var num_dt_chambers=0;

var fIter = 'iterN'

var CSC_TYPES = [
	['ME+','p',[
		['ME+4','4',[
			['ME+4/ALL','ALL', 36],
			['ME+4/2','2', 36],
			['ME+4/1','1', 18]]
		],
		['ME+3','3',[
			['ME+3/ALL','ALL', 36],
			['ME+3/2','2', 36],
			['ME+3/1','1', 18]]
		],
		['ME+2','2',[
			['ME+2/ALL','ALL', 36],
			['ME+2/2','2', 36],
			['ME+2/1','1', 18]]
		],
		['ME+1','1',[
			['ME+1/ALL','ALL', 36],
			['ME+1/3','3', 36],
			['ME+1/2','2', 36],
			['ME+1/1','1', 36]
			//['ME+1/4','4', 36]
			]
		]]
	],
	['ME-','m',[
		['ME-1','1',[
			//['ME-1/4','4', 36],
			['ME-1/1','1', 36],
			['ME-1/2','2', 36],
			['ME-1/3','3', 36],
			['ME-1/ALL','ALL', 36]]
		],
		['ME-2','2',[
			['ME-2/1','1', 18],
			['ME-2/2','2', 36],
			['ME-2/ALL','ALL', 36]]
		],
		['ME-3','3',[
			['ME-3/1','1', 18],
			['ME-3/2','2', 36],
			['ME-3/ALL','ALL', 36]]
		],
		['ME-4','4',[
			['ME-4/1','1', 18],
			['ME-4/2','2', 36],
			['ME-4/ALL','ALL', 36]]
		]]
	]
];

var CSC_TYPES_ORG = [
	['ME+',[
		["ME+4/2", 36],
		["ME+4/1", 18],
		["ME+3/2", 36],
		["ME+3/1", 18],
		["ME+2/2", 36],
		["ME+2/1", 18],
		["ME+1/3", 36],
		["ME+1/2", 36],
		["ME+1/1", 36]]
	],
	['ME-',[["ME-1/1", 36],
		["ME-1/2", 36],
		["ME-1/3", 36],
		["ME-2/1", 18],
		["ME-2/2", 36],
		["ME-3/1", 18],
		["ME-3/2", 36],
		["ME-4/1", 18],
		["ME-4/2", 36]]
	]
];

var DT_TYPES = [
	['MB+2','2',[
		['MB+2/1','1', 12],
		['MB+2/2','2', 12],
		['MB+2/3','3', 12],
		['MB+2/4','4', 14]]
	],
	['MB+1','1',[
		['MB+1/1','1', 12],
		['MB+1/2','2', 12],
		['MB+1/3','3', 12],
		['MB+1/4','4', 14]]
	],
	['MB-0','0',[
		['MB-0/1','1', 12],
		['MB-0/2','2', 12],
		['MB-0/3','3', 12],
		['MB-0/4','4', 14]]
	],
	['MB-1','-1',[
		['MB-1/1','1', 12],
		['MB-1/2','2', 12],
		['MB-1/3','3', 12],
		['MB-1/4','4', 14]]
	],
	['MB-2','-2',[
		['MB-2/1','1', 12],
		['MB-2/2','2', 12],
		['MB-2/3','3', 12],
		['MB-2/4','4', 14]]
	],
	['MB-ALL','ALL',[
		['MB-ALL/1','1', 12],
		['MB-ALL/2','2', 12],
		['MB-ALL/3','3', 12],
		['MB-ALL/4','4', 14]]
	]
];

var DQM_SEVERITY = 
	[
	{"name": "NONE", "color": "lightgreen", "hex":"#90EE90"},
	{"name": "UNCERT05", "color": "lightgreen", "hex":"#94E26f"},
	{"name": "UNCERT075", "color": "lightgreen", "hex":"#96D953"},
	{"name": "UNCERT1", "color": "yellowgreen", "hex":"#9ACD32"},
	{"name": "LOWSTAT", "color": "yellow", "hex":"#FFFF00"},
	{"name": "TOLERABLE", "color": "lightpink", "hex":"#FFB6C1"},
	{"name": "SEVERE", "color": "orange", "hex":"#FFA500"},
	{"name": "CRITICAL", "color": "red", "hex":"#FF0000"}
	];

/*
 * var DQM_SEVERITY = ["NONE"], ["MINOR"], ["TOLERABLE"], ["SEVERE"],
 * ["CRITICAL"] ];
 */


var re_me_c = /ME(\+|-)(\d)\/(\d|ALL)\/(\d\d)/;
var re_me_r = /ME(\+|-)(\d)\/(\d|ALL)/;
var re_me_s = /ME(\+|-)(\d)/;
var re_me_e = /ME(\+|-)/;

var re_mb_c = /MB([\+-]\d|ALL)\/(\d)\/(\d\d)/;
var re_mb_s = /MB([\+-]\d|ALL)\/(\d)/;
var re_mb_w = /MB([\+-]\d|ALL)/;


function idToDir(id)
{
	var ssystem = id.substr(0,2);
	if (ssystem == "MB") return "MB/" + id + "/";
	if (ssystem == "ME") return "ME" + id.substr(2,1) + "/" + id.substr(3) + "/";
	return ""
}

function idToWheel(id)
{
	if (id.substr(0,2)!="MB") return null;
	m = re_mb_w.exec(id);
	if (m!=null) return m[1]
	return null;
}

function idToEndcap(id)
{
	if (id.substr(0,2)!="ME") return null;
	res = id.substr(2,1)
	if (res=="+" || res=="-") return res;
	return null;
}

function idToStation(id)
{
	if (id.substr(0,2)=="MB") {
		m = re_mb_s.exec(id);
		if (m!=null) return m[2]
	}
	if (id.substr(0,2)=="ME") {
		m = re_me_s.exec(id);
		if (m!=null) return m[2]
	}
	return null;
}

function idToRing(id)
{
	if (id.substr(0,2)!="ME") return null;
	m = re_me_r.exec(id);
	if (m!=null) return m[3]
	return null;
}

function idToChamber(id)
{
	if (id.substr(0,2)=="MB") {
		m = re_mb_c.exec(id);
		if (m!=null) return m[3]
	}
	if (id.substr(0,2)=="ME") {
		m = re_me_c.exec(id);
		if (m!=null) return m[4]
	}
	return null;
}


function showHide(id)
{
	obj = document.getElementById(id);
	if (obj) {
		if (obj.style.display == 'none') obj.style.display = '';
		else obj.style.display = 'none';
	}
	return false;
}


function showId(id)
{
	obj = document.getElementById(id);
	if (obj) obj.style.display = '';
	return false;
}


function hideId(id)
{
	obj = document.getElementById(id);
	if (obj) obj.style.display = 'none';
	return false;
}


function addPadding(num,count,pad)
{
	var numStr = num + '';
	while(numStr.length < count) {
	numStr = pad + numStr;
	}
	return numStr;
}


function pad0X(num)
{
	// '0' padding is added only if num is single digit
	if (num>=0 && num <10) return '0'+num;
	return num;
}


function stripLeadingZeroes(st)
{
	idx=0;
	len=st.length;
	while (st[idx]==0 && idx<len-1) idx++;
	return st.substring(idx);
}


function getTime()
{
	var t = new Date();
	val = t.getDate()
	date = (val<10)?'0'+val:val;
	val = t.getMonth()+1;
	month = (val<10)?'0'+val:val;
	year = t.getFullYear();	
	val = t.getHours();
	hour = (val<10)?'0'+val:val;
	val = t.getMinutes();
	minute = (val<10)?'0'+val:val;
	val = t.getSeconds();
	sec = (val<10)?'0'+val:val;
	time_str = date+"/"+month+"/"+year+" "+hour+":"+minute+":"+sec;
	return time_str;
}

function addlog(message, level)
{
	if (Log) {
		var entry = new Array();
		entry[0] = getTime();
		entry[1] = level;
		entry[2] = message;
		Log.push(entry);
		showLog();
	}
/*
 * var logd = document.getElementById("log"); if (logd) { log_text =
 * logd.innerHTML;
 * 
 * msg = '<tr><td class="log">'+getTime()+'</td><td><span class="log"
 * id=\"'+level+'\">'+level.toUpperCase()+ '</span></td><td><span
 * class="log">' +message+'</span></td></tr><br>' + log_text; //msg = '<li class="log" id=\"'+level+'\">'+level.toUpperCase()+ ": "
 * +message+'</li>' + log_text; logd.innerHTML = msg; } return false;
 */
}


function showLog()
{
	var logd = document.getElementById("log");
	if (Log && logd && !isIE) {
		logd.innerHTML = "";
		log_out = "<table class='log'>";
		if (Log.length >0) log_out += "<tr><th>Date/Time</th><th>Level</th><th>Message</th></tr>";
		for (var i=Log.length-1; i>=0; i--) {
			if (Log[i] && Log[i].length==3) {
				level = Log[i][1];
				if (level == logLevel || logLevel == "all") {
					log_out += '<tr><td class="log">'+Log[i][0]+'</td><td class="log"><span class="log" id=\"'+Log[i][1]+'\">'+Log[i][1].toUpperCase()+
						'</span></td><td width="100%" class="log">' +Log[i][2]+'</td></tr>';
				}
			}
		}
		log_out +="</table>";
		logd.innerHTML = log_out;
	}
}


function selectLogLevel()
{
	var selObj = document.getElementById("selLogLevel");
	if (selObj) {
		var selIdx = selObj.selectedIndex;
		selList = selObj.options[selIdx].value;
		logLevel = selList;
		// addlog(selList, "info");
		showLog();
	}
}


function clearLog() 
{
	Log.length = 0;
	showLog();
}


function showTestHelp(test)
{
	var help_url="http://www.phys.ufl.edu/cms/emu/dqm";
	try {
		var generator=window.open('test_help','name','height=600,width=600,resizable=true');
		if (window.focus) {generator.focus()}
		generator.document.write('<html><head><title>Help page for '+test+'</title>');
		generator.document.write("<style>body {color: #000000; background-color: #9090FF;margin: 2px;font-family:Arial;	overflow: auto;}"+
		"span.winTitle {	margin: 0;font-size: 12px;font-weight: bold;text-decoration: none;color: #000000;}"+
		"fieldset {background-color: #ddd;	border-style: solid;border-width: 1px;border-color: black;margin: 2px;padding: 2px;}"+
		"legend {border-style: solid;border-width: 1px;background-color:  #C0C0FF;}</style>");
		generator.document.write('</head><body>');
		generator.document.write("<fieldset><legend><span class='winTitle'>Help page for "+test+"</span></legend>");
		generator.document.write("</fieldset>");
		// generator.document.write("<div id='help_div'></div>");
		generator.document.write("<iframe width=100% height=100% id='help_div' src='"+help_url+"'></iframe>");
		
		// generator.document.write('<p><a
		// href="javascript:self.close()">Close</a> the popup.</p>');
		generator.document.write('</body></html>');
		generator.document.close();
	  } 
	  catch (exc){
	}

}


function showPlot() {
	var imgurl = ResultsFolder+parent.RunNumber+postfix+"/"+fIter+"/"+Folder+"/"+Canvas;
	var ref_imgurl = refURL+"/"+Folder+"/"+Canvas;
	var out = "";
	var title = "Plots";
	if (!isFolderValid) {
		// out += RunNumber+": Invalid folder - \""+Folder+"\"<hr>";
	} else {
		if ((parent.Folder != "") && (parent.Canvas != "") ) {
			if ((Folder.search("ME") >=0 && Scope == "CSC") || 
				(Folder.search("MB") >=0 && Scope == "DT") || 
				(Folder.search("common") >=0 && parent.Scope == "ALL")) {
				// out += RunNumber+": \""+Folder+"/"+Canvas+"\"<hr>";
				title += ": " +RunNumber+"/" + fIter + "/" +Folder+"/"+Canvas;
				// addlog(title, "info")
				// out += "<a target=blank href=\'"+imgurl+"\'><IMAGE
				// height='"+imgHeight+"' NAME='test' SRC='"+imgurl+"'></a>";
				// dirtypop();
				if (fOpenExternal) {
					try {
						var generator=window.open('plots','name','height=620,width=800,resizable=true');
						if (window.focus) {generator.focus()}
						generator.document.write('<html><head><title>'+title+'</title>');
						generator.document.write("<style>body {color: #000000; background-color: #9090FF;margin: 2px;font-family:Arial;	overflow: auto;}"+
						"span.winTitle {	margin: 0;font-size: 12px;font-weight: bold;text-decoration: none;color: #000000;}"+
						"fieldset {background-color: #ddd;	border-style: solid;border-width: 1px;border-color: black;margin: 2px;padding: 2px;}"+
						"legend {border-style: solid;border-width: 1px;background-color:  #C0C0FF;}</style>");
						generator.document.write('</head><body>');
						generator.document.write("<fieldset><legend><span class='winTitle'>"+title+"</span></legend>");
						
						generator.document.write("<IMAGE width=100% NAME='test' SRC='"+imgurl+"'>");
						generator.document.write("</fieldset>");
						// generator.document.write('<p><a
						// href="javascript:self.close()">Close</a> the
						// popup.</p>');
						generator.document.write('</body></html>');
						generator.document.close();
					} 
					catch (exc){
					}
				} else {
					out += "<a target=blank href=\'"+imgurl+"\'><IMAGE height='"+imgHeight+"' NAME='test' SRC='"+imgurl+"'></a>";
				}
				
				if (fShowReference) {
					try {
						var generator=window.open('ref_plots','ref_name','height=640,width=800,resizable=true');
						if (window.focus) {generator.focus()}
						var ref_title = "Reference Plots: " +Folder+"/"+Canvas;
						generator.document.write('<html><head><title>'+ref_title+'</title>');
						generator.document.write("<style>body {color: #000000; background-color: #90FF90;margin: 2px;font-family:Arial;	overflow: auto;}"+
						"span.winTitle {	margin: 0;font-size: 12px;font-weight: bold;text-decoration: none;color: #000000;}"+
						"fieldset {background-color: #ddd;	border-style: solid;border-width: 1px;border-color: black;margin: 2px;padding: 2px;}"+
						"legend {border-style: solid;border-width: 1px;background-color:  #C0FFC0;}</style>");
						generator.document.write('</head><body>');
						generator.document.write("<fieldset><legend><span class='winTitle'>"+ref_title+"</span></legend>");
						generator.document.write("<span style='color: #FF0000'><blink>Warning: These Reference Plots are not certified yet!!!</blink></span><br>");
						generator.document.write("<IMAGE width=100% NAME='test' SRC='"+ref_imgurl+"'>");
						generator.document.write("</fieldset>");
						// generator.document.write('<p><a
						// href="javascript:self.close()">Close</a> the
						// popup.</p>');
						generator.document.write('</body></html>');
						generator.document.close();
					} 
					catch (exc){
					}
					
				}
				/*
				else {
					out += "<a target=blank href=\'"+ref_imgurl+"\'><IMAGE height='"+imgHeight+"' NAME='test' SRC='"+ref_imgurl+"'></a>";
				}
				*/
			}else {
				// addlog("Invalid canvas: \""+RunNumber+"/"+Folder+"/"+Canvas, "error");
				addlog("Can not display '"+Canvas+"'. Selected canvas scope '"+Scope+"' and folder '"+Folder+"' logical sections mismatch", "warn");
				// out += "Invalid canvas: \""+Folder+"/"+Canvas+"\"<hr>";
			}
		}
	}
	var o_obj = document.getElementById("plots_div");
	if (o_obj) o_obj.innerHTML = out;
	o_obj = document.getElementById("plot_title");
	if (o_obj) o_obj.innerHTML = title;
	return false;
}


function showCSCTable(id)
{
	// DDUs.length = 0;
	// clearShowList("Folders", "All CSCs");
	var c_list = CSC_TYPES;
	var link = ResultsFolder+RunNumber+postfix;
	var rootfile_link = ResultsFolder+RunNumber+".root";
	// genDDUMap();
	clearShowList("Folders", "All CSCs");
	obj = document.getElementById(id);
	if (obj) {
		if (c_list && c_list.length > 0) {
			// obj.innerHTML = "";
			out = "";
			out += "<table border=1 cellpadding=0 cellspacing=0 id='csc_table'>";
			// out += "<th class='RunLink' id='RunLink' colspan=38
			// align=left>Run Number: </th>";
			out += "<th class='RunLink' align=left colspan=39>Run Number: <a class='RLink' target=images href="+link+">"+RunNumber+
				"</a> [<a href="+rootfile_link+">ROOT file</a>]<br></th>";
			// Display Commons
			out += "<tr class='me'><td id='cscCommon' class='me_emu' colspan='2' >EMU Summary</td><td colspan='37'></td></tr>";
			
			// Display CSCs
			for (var i=0; i< c_list.length; i++) 
			{
				if (c_list[i] && c_list[i].length > 0) {
					var side = c_list[i][0];
					var stations =  c_list[i][2];
					out += "<tr class='me'><td rowspan='14' id='"+side+"' class='me' >"+side+"</td>";
					if (stations && stations.length > 0) { // first endcap's all rings first
						for (var s=0; s< stations.length; s++) {
							if (side=='ME-') break;
							for (var s=0; s< stations.length; s++) {
								var station = stations[s][0];
								var types = stations[s][2];
								if (types && types.length > 0) {
									var k=0;
									for (var j=0; j< types.length; j++) {
										if (types[j][1]!='ALL') continue;
										var csctype = types[j][0];
										if (k>0) out+= "<tr class='me_type'>";
										out += "<td colspan=2 id='"+csctype+"' class='me_type' >"+csctype+"</td>";
										var num_cscs = types[j][2];
										for (var k=1; k<= num_cscs; k++) {
											out += "<td class='me_csc' id='"+(csctype+"/"+pad0X(k))+"'>"+k+"</td>";
										}
										if (k>0) out += "</tr>";
										k += 1;
									}
								}
							}
							//addlog("<pre>"+out+"</pre>","debug");
						}
						if (side=='ME+') out += "<tr><td  colspan=38 style=\"border:2px\"></td></tr>";
						for (var s=0; s< stations.length; s++) {
							var station = stations[s][0];
							var types = stations[s][2];
							if (s>0) out+= "<tr class='me_station'>";
							out += "<td rowspan='"+(types.length-1)+"' id='"+station+"' class='me_station' >"+station+"</td>";
							var k=0;
							if (types && types.length > 0) {
							    for (var j=0; j< types.length; j++) {
								    if (types[j][1]=='ALL') continue;
									var csctype = types[j][0];
									if (k>0) out+= "<tr class='me_type'>";
									out += "<td id='"+csctype+"' class='me_type' >"+csctype+"</td>";
									var num_cscs = types[j][2];
									for (var k=1; k<= num_cscs; k++) {
										if (num_cscs==18)	out += "<td class='me_csc' id='"+(csctype+"/"+pad0X(k))+"' colspan='2'>"+k+"</td>";
										else				out += "<td class='me_csc' id='"+(csctype+"/"+pad0X(k))+"'>"+k+"</td>";
									}
									if (k>0) out += "</tr>";
									k += 1;
								}
							}
							if (s>0) out += "</tr>";
						}
						if (side=='ME-') out += "<tr><td colspan=38 style=\"border:2px\"></td></tr>";
						for (var s=0; s< stations.length; s++) {  // second endcap's all rings last
							if (side=='ME+') break;
							for (var s=0; s< stations.length; s++) {
								var station = stations[s][0];
								var types = stations[s][2];
								if (types && types.length > 0) {
									var k=0;
									for (var j=0; j< types.length; j++) {
										if (types[j][1]!='ALL') continue;
										var csctype = types[j][0];
										if (k>0) out+= "<tr class='me_type'>";
										out += "<td colspan=2 id='"+csctype+"' class='me_type' >"+csctype+"</td>";
										var num_cscs = types[j][2];
										for (var k=1; k<= num_cscs; k++) {
											out += "<td class='me_csc' id='"+(csctype+"/"+pad0X(k))+"'>"+k+"</td>";
										}
										if (k>0) out += "</tr>";
										k += 1;
									}
								}
							}
							//addlog("<pre>"+out+"</pre>","debug");
						}
					}
					out += "</tr>";
				}
			}
			out += "</table>";
			out += printReportLegend();
			obj.innerHTML = out;
		}
	}
}

function showDTTable(id)
{
	// DDUs.length = 0;
	// clearShowList("Folders", "All DTs");
	var c_list = DT_TYPES;
	var link = ResultsFolder+RunNumber+postfix;
	var rootfile_link = ResultsFolder+RunNumber+".root";
	// genDDUMap();
	clearShowList("Folders", "All DTs");
	obj = document.getElementById(id);
	if (obj) {
		if (c_list && c_list.length > 0) {
			// obj.innerHTML = "";
			out = "";
			out += "<table border=1 cellpadding=0 cellspacing=0 id='dt_table'>";
			// out += "<th class='RunLink' id='RunLink' colspan=38
			// align=left>Run Number: </th>";
			out += "<th class='RunLink' align=left colspan=16>Run Number: <a class='RLink' target=images href="+link+">"+RunNumber+
				"</a> [<a href="+rootfile_link+">ROOT file</a>]<br></th>";
			// Display Commons
			out += "<tr class='me'><td id='dtCommon' class='me_emu' colspan='2' >DT Summary</td><td colspan='14'></td></tr>";
			
			// Display CSCs
			for (var i=0; i< c_list.length; i++) 
			{
				if (c_list[i] && c_list[i].length > 0) {
					var side = c_list[i][0];
			 		out += "<tr class='me'><td rowspan='4' id='"+side+"' class='me' >"+side+"</td>";
					var types =  c_list[i][2];
					if (types && types.length > 0) {
						for (var j=0; j< types.length; j++) {			
							var dttype = types[j][0];
							if (j>0) out+= "<tr class='me_type'>";
							out += "<td id='"+dttype+"' class='me_type' >"+dttype+"</td>";		
							var num_dts = types[j][2];
							for (var k=1; k<= num_dts; k++) {
								if (num_dts==12) {
									if (k==4 || k==10)	out += "<td class='me_csc' id='"+(dttype+"/"+pad0X(k))+"' colspan='2'>"+k+"</td>";
									else				out += "<td class='me_csc' id='"+(dttype+"/"+pad0X(k))+"'>"+k+"</td>";
								}
								if (num_dts==14) {
									if (k<4)		out += "<td class='me_csc' id='"+(dttype+"/"+pad0X(k))+"'>"+k+"</td>";
									else if (k==4)	out += "<td class='me_csc' id='"+(dttype+"/13")+"'>"+13+"</td>";
									else if (k<11)	out += "<td class='me_csc' id='"+(dttype+"/"+pad0X(k-1))+"'>"+(k-1)+"</td>";
									else if (k==11)	out += "<td class='me_csc' id='"+(dttype+"/14")+"'>"+14+"</td>";
									else			out += "<td class='me_csc' id='"+(dttype+"/"+pad0X(k-2))+"'>"+(k-2)+"</td>";
								}
							}
							if (j>0) out += "</tr>";
						}
					}
					out += "</tr>";		
				}
			}
			out += "</table>";
			out += printReportLegend();
			obj.innerHTML = out;
			// alert(out);
		}
	}
}


function getVMECrateName(crate)
{
	if (VMEMAP && VMEMAP.length) {
		for (var i=0; i<VMEMAP.length; i++) {
			if (crate == "crate"+VMEMAP[i][1]) {
				return VMEMAP[i][0];
			}
		}
	}
}


function showCSCCounters()  
{
	cntrs=CSC_COUNTERS;
	//csc_map = CSCMAP;
	csc_map=[];
	var obj=document.getElementById("csccounters_div");
	if (obj) {
		var out="<table>";
		var crate=-1;
		var slot=-1;
		thresh_alct=25;
		thresh_clct=20;
		thresh_cfeb=50;
		thresh_bad=10;
		cnt_bad_alct=0;
		cnt_bad_clct=0;
		cnt_bad_cfeb=0;
		cnt_bad_events=0;
		
		if (cntrs && cntrs.length >0) {
			cscs = cntrs[0];

			for (var i=1; i<cscs.length; i++) {
				if (!cscs[i]) continue;
				var csc = cscs[i][0];	
				stats = cscs[i][1];
				n_crate = parseFloat(csc.substring(4,7));
				if (n_crate != crate ) { 
					crate_str=getVMECrateName("crate"+n_crate) + ": crate"+n_crate;
					out+="</tr><tr><td><b>"+crate_str+"</b></td>";
					crate=n_crate;
				}
				slot = parseFloat(csc.substring(8,10));
				cscid = "";
				for (var k=0; k<csc_map.length; k++) {
						if (!csc_map[k]) continue;
						map_entry = csc_map[k];
						if ((map_entry[0] == "crate"+crate) && (map_entry[1] == "slot"+slot)) {cscid = map_entry[2];}
				}
				stats_out="";
				alct=0;
				clct=0;
				cfeb=0;
				bad=0;
				dmb=0;
				if (stats && stats.length) {
					for (var j=0; j<stats.length; j++) {
						// stats_out+= stats[j][1]+",";
						tag = stats[j][0];
						if (tag=='ALCT') alct=parseInt(stats[j][1]);
						if (tag=='CLCT') clct=parseInt(stats[j][1]);
						if (tag=='CFEB') cfeb=parseInt(stats[j][1]);
						if (tag=='BAD') bad=parseInt(stats[j][1]);
						if (tag=='DMB') dmb=parseFloat(stats[j][1]);
					}
				}
				if (dmb) {
					bcolor='white';
					color="black";
					badclr='#FFD0D0';
					if ((100*alct/(dmb))<thresh_alct) { color="red"; cnt_bad_alct++;}
					stats_out+="A:<font color='"+color+"'>"+(100*(alct/(dmb))).toFixed(1)+"</font>;";
					color="black";
					if ((100*clct/(dmb))<thresh_clct) { color="red"; cnt_bad_clct++;}
					stats_out+="T:<font color='"+color+"'>"+(100*(clct/(dmb))).toFixed(1)+"</font>;";
					color="black";
					if ((100*cfeb/(dmb))<thresh_cfeb) { color="red"; cnt_bad_cfeb++;}
					stats_out+="C:<font color='"+color+"'>"+(100*(cfeb/(dmb))).toFixed(1)+"</font>";// ;Bad:"+bad;
				
					color="black";
					if ((100*bad/dmb)>thresh_bad) { color="red"; bcolor=badclr; cnt_bad_events++;}
					out += "<td style='background-color: "+bcolor+"'><b>(dmb"+slot+") "+cscid+"</b><br>#E:"+dmb+";"					
					out +=" #B:<font color='"+color+"'>"+bad+"("+(100*(bad/(dmb))).toFixed(1)+")</font><br>"+stats_out+"</td>";
				}
				// slots = crates[i][1];
				out += "";
				// addlog(csc, "debug");
			}		
		}		
		out+= "</table>";	
		out= "<table><tr><td class='node_info'>Summary (#CSCs with failed efficiencies)</td>"+
		"<td class='node_info'>Bad events: <b>"+cnt_bad_events+"</b></td><td class='node_info'>ALCT: <b>"+cnt_bad_alct+"</b></td>"+
		"<td  class='node_info'>TMB/CLCT: <b>"+cnt_bad_clct+"</b></td><td  class='node_info'>CFEB: <b>"+cnt_bad_cfeb+"</b></td></tr></table>"+
		"<table><tr><td class='node_info'>Legend: #E - #of Events; #B - #of Bad Events; A - ALCT; T - TMB/CLCT; C - CFEB Efficiencies in %</td></tr>"+
		"<tr><td class='node_info'>Fail thresholds: Bad Events:>"+thresh_bad+
		"%; ALCT:<"+thresh_alct+"%; CLCT:<"+thresh_clct+"%; CFEB:<"+thresh_cfeb+"%</td></tr></table>" + out;
		obj.innerHTML=out;
		return;
	}
}

function getCSCCounters(id)
{
	var fullUrl = ResultsFolder+RunNumber+".plots/" + CSCCounters;
	
	var req=false;
	if (window.XMLHttpRequest) {
		req = new XMLHttpRequest();
	} else if (window.ActiveXObject) {
		// req = new ActiveXObject("Microsoft.XMLDOM");
		req = new ActiveXObject((navigator.userAgent.toLowerCase().indexOf('msie 5') != -1) ? "Microsoft.XMLHTTP" : "Msxml2.XMLHTTP");
		// var req = new ActiveXObject("Microsoft.XMLHTTP");
	}
	req.open("GET",fullUrl,true); // true= asynch, false=wait until loaded
	
	req.onreadystatechange = function() {
		if (req.readyState == 4) {
			if (req.status==200) {
				var reply = req.responseText;
				// Need to strip malformed array string to satisfy IE7
				reply = reply.replace(/,]/g,"]");
				reply = reply.replace("var ","");
				window.eval(reply);
				showCSCCounters();
				// addlog("Run " + RunNumber + ": Found "+num_chambers+" CSCs", "info");
				// addlog(reply, 'debug');
				// updateBrowserPage(id,false,2);
			} else if (req.status==404) {
				addlog("Can not load CSC Counters " + fullUrl, 'warn');
				CSC_COUNTERS=false;
				showCSCCounters();
			}
		}
	}
	req.send(null);
	return false;
}


// Load Available offline Runs list
function showRunsList(r_list) 
{
	out = "<table class='runs_table'>";
	out += "<th>Run</th><th>Analyzed on</th>";
	cnt=0;
	if (r_list) {
		for (var i = 0; i< r_list.length; i++) {
			child = [];
			child=r_list[i];
			if (!child) continue;
			run_descr = child[0].replace(/.plots/g,'');
			run_num = child[0].substr(0, child[0].search(/.plots/g));
			run_time = child[0].substr(child[0].search(/\(/g),child[0].search(/\)/g));
			if (i==0) RunNumber = run_num;
			// addlog(run_descr, "info");
			cnt++;
			//out += "<tr><td><a href='' id='"+run_num+"' onClick='RunNumber=\""+run_num+
			//	"\";getTestsList();getCSCCounters();getSystemsList(\""+run_num+
			//	"\");return false;'>"+run_num+"</a></td><td>"+run_time+"</td></tr>";
			out += "<tr><td><a href='' id='"+run_num+
				"' onClick='RunNumber=\""+run_num+"\";getTestsList();getSystemsList(\""+run_num+"\");return false;'>"
				+run_num+"</a></td><td>"+run_time+"</td></tr>";
		}
	}
	if (cnt==0) {
		out += "<tr><td>No Run List info</td></tr>";
	}
	out += "</table>";
	obj = document.getElementById("runs_list");
	if (obj) obj.innerHTML = out;
}


function getRunsList()
{
	var fullUrl = ResultsFolder + RunsList;
	
	
	var req=false;
	if (window.XMLHttpRequest) { 
		/*
		 * try {
		 * netscape.security.PrivilegeManager.enablePrivilege("UniversalBrowserRead
		 * UniversalBrowserWrite"); } catch (e) { alert("Permission
		 * UniversalBrowserRead denied."); }
		 */
		req = new XMLHttpRequest();
	} 
	else if (window.ActiveXObject) {
		// req = new ActiveXObject("Microsoft.XMLDOM");
		req = new ActiveXObject((navigator.userAgent.toLowerCase().indexOf('msie 5') != -1) ? "Microsoft.XMLHTTP" : "Msxml2.XMLHTTP");
		// var req = new ActiveXObject("Microsoft.XMLHTTP");
	}
	req.open("GET",fullUrl,true); // true= asynch, false=wait until loaded
	
	req.onreadystatechange = function() {
		if (req.readyState == 4) {
			if (req.status==200) {
				var reply = req.responseText;
				// Need to strip malformed array string to satisfy IE7
				reply = reply.replace(/,]/g,"]");
				reply = reply.replace("var ","");
				window.eval(reply);
				showRunsList(RUNS);
				// addlog(reply, 'debug');
				// updateBrowserPage(Canvas,false,3);
				// updateBrowserPage(id,false,2);
			} else if (req.status==404) {
				addlog("Can not load " + fullUrl, 'error');
				RUNS=false;
				showRunsList(RUNS);
			}
		}
	}
	req.send(null);	
	return false;
}


//Load Available Tests/Canvases list
function showTestsList(r_list, id) 
{
	obj = document.getElementById(id);
	if (obj && r_list) {
		out = "";
		for (var i = 0; i< r_list.length; i++) {
			// document.write("<tr><td><b>"+r_list[i][0]+"</b></td></tr>");
			if (!r_list[i]) continue;
			out += "<fieldset>";
			th_folder = r_list[i][0];
			th_id = "folder_"+ th_folder;
			out += "<legend class='t_hdr'><a class='test_hdr' href='#' onClick='showHide(\""+th_id+"\");'>"+th_folder+"</a></legend>";

			if (r_list[i][0] == "CSC")		scope = "CSC";
			else if (r_list[i][0] == "DT")	scope = "DT";
			else scope = "ALL";

			var child = [];
			child=r_list[i];
			if (!child) continue;
			if (child.length <=2) continue;
			
			out += "<div class='t_div' id='"+th_id+"' style='overflow: hidden;'>";
			t_map = [];
			for (var j=2; j<child.length; j++) {
				if (!child[j]) continue;
				t_id = child[j][1];
				t_name = child[j][0];
				// === Replace with search()
				if (t_name.lastIndexOf(":") != -1) {
					t_folder = t_name.substr(0, t_name.lastIndexOf(":"));
					t_name = t_name.substr(t_name.lastIndexOf(':')+1, t_name.length);
				} else {
					t_folder = th_folder;
				}

				if (!t_map[t_folder])
					t_map[t_folder] = new Array();
				if (!t_map[t_folder][t_id])
					t_map[t_folder][t_id] = t_name;
			}
			for (var list in t_map) {
				if ((list != th_folder) || (list == "CSC") || (list == "DT")) {
					out += "<fieldset>";
					out += "<legend class='t_hdr'><a class='test_hdr2' href='#' onClick='showHide(\"id_"+list+"\");'>"+list+"</a></legend>";
					out += "<div class='t_div' id='id_"+list+"' style='overflow: hidden;'>";
				}
				out += "<table class='tests_table'>";
				folder = t_map[list];
				for (var entry in folder) {
					// <td><input type=checkbox id='cb_"+entry+"'></td>
					// <a class='test_help' href='' onClick='showTestHelp(\""+folder[entry]+"\"); return false;'>[?]</a>
					out += "<tr><td id =\""+entry+"\"><a class='add_to_list' href='' "+
						"onClick='addToTestsShowList(\"Custom\",\""+entry+"\",\""+folder[entry]+"\" ,\""+scope+"\");return false;'>[+]</a>"+
						"</a><a class='test_item' href='' "+
						"onClick='Canvas=\""+entry+"\";Scope=\""+scope+"\";return updateBrowserPage(\""+entry+"\",false,3)'>"+
					folder[entry]+"</a></td></tr>";
					if (scope=="CSC" ) addToTestsShowList("All CSC Tests", entry, folder[entry], scope);
					if (scope=="DT" ) addToTestsShowList("All DT Tests", entry, folder[entry], scope);
				}
				out += "</table>";

				if ((list != th_folder)|| (list == "CSC") || (list == "DT")) out += "</div></fieldset>";
			}
			out += "</div></fieldset>";
		}
		obj.style.display = "";
		obj.innerHTML = out;
		hideId("folder_DT");
	}
}


function getTestsList()
{
	var fullUrl = ResultsFolder+RunNumber+".plots/" + TestsList;
	
	var req=false;
	if (window.XMLHttpRequest) { 
		req = new XMLHttpRequest();
	}
	else if (window.ActiveXObject) {
		req = new ActiveXObject((navigator.userAgent.toLowerCase().indexOf('msie 5') != -1) ? "Microsoft.XMLHTTP" : "Msxml2.XMLHTTP");
	}
	req.open("GET",fullUrl,true); // true= asynch, false=wait until loaded
	
	req.onreadystatechange = function() {
		if (req.readyState == 4) {
			if (req.status==200) {
				var reply = req.responseText;
				// Need to strip malformed array string to satisfy IE7
				reply = reply.replace(/,]/g,"]");
				reply = reply.replace("var ","");
				window.eval(reply);
				showTestsList (CANVASES_LIST, "tests_div");
				// addlog(reply, 'debug');
				updateBrowserPage(Canvas,false,3);				
				// updateBrowserPage(id,false,2);
			} else if (req.status==404) {
				addlog("Can not load " + fullUrl, 'error');
			}
		}
	}
	req.send(null);	
	return false;
}


function showTextReport(id)
{
	var obj = document.getElementById(id);
	if (obj) {
		var fullUrl = ResultsFolder+RunNumber+".plots/" + TextReport;
		if (reportWindow && !reportWindow.closed) openReportInWindow();
		// out = "<b>DQM Summary Report for Run: <a target='_blank' href='"
		// +fullUrl+"'>" + RunNumber + "</a></b><br>";
		out = "DQM Summary Report for Run: <b><a href='" + fullUrl + "'>" + RunNumber + "</a> <a href='' onClick='openReportInWindow();return false;'>[Open]</a></b><br>";
		out += "<pre>";
		out += DQMReport;
		out += "</pre>";
		obj.innerHTML = out;
	}
}


function getTextReport()
{
	var fullUrl = ResultsFolder+RunNumber+".plots/" + TextReport;
	var req=false;
	if (window.XMLHttpRequest) {
		req = new XMLHttpRequest();
	} else if (window.ActiveXObject) {
		req = new ActiveXObject((navigator.userAgent.toLowerCase().indexOf('msie 5') != -1) ? "Microsoft.XMLHTTP" : "Msxml2.XMLHTTP");
	}
	req.open("GET",fullUrl,true); // true= asynch, false=wait until loaded
	
	req.onreadystatechange = function() {
		if (req.readyState == 4) {
			if (req.status==200) {
				DQMReport = req.responseText;
				showTextReport("report_div");
			} else if (req.status==404) {
				DQMReport = "No Summary Report"
				addlog("Can not load " + fullUrl, 'warn');
				showTextReport("report_div");
			}
		}
	}
	req.send(null);
	return false;
}


function showDQMReport(id)
{
	obj = document.getElementById(id);
	if (obj) {
		if (DQM_REPORT) {
			if (reportWindow && !reportWindow.closed) openReportInWindow();
			var fullUrl = ResultsFolder+RunNumber+".plots/" + TextReport;	
			// var out="<b>DQM Summary Report for Run: <a href='"+fullUrl+"'>"+DQM_REPORT.run+"</a><br> (generated on "+DQM_REPORT.genDate+")</b>";
			// out += <a href='' onClick='openReportInWindow();return false;'>[Open]</a></h5>";
			var out ="";
			out += printReportLegend();
			out += "<table class='dqm_report'>";
			// addlog("DQM report #entries: " + DQM_REPORT.report.length,"info");
			var report = DQM_REPORT.report;
			if (report.length) {
				for (var i = 0; i < report.length; i++) {
					if (report[i]) {
						var obj_sev = 0;
						out += "<tr><th class='repScope' colspan=3><b>" + report[i].name + "</b></th></tr>";
						if (report[i].list && report[i].list.length) {
							err_list = report[i].list;
							for (var j=0; j< err_list.length; j++) {
								var entry = err_list[j];
								if (entry) {
									// out += "<tr><td></td><td>" + entry.descr + "</td><td>"+entry.severity+"</td></tr>";
									var sev_idx = parseInt(entry.severity);
									var sev_name = DQM_SEVERITY[sev_idx].name;
									var sev_str = "";
									if ( sev_idx > 0) {
										sev_str = "<span>" + sev_name + "</span>";
									}
									if (sev_idx > obj_sev) obj_sev = sev_idx;
									var descr_str = entry.descr;
									if (entry.testID != "") {
										var testInfo = findTestPlotInfo(entry.testID);
										if (testInfo) {
											var testFolder = getFolderName(report[i].objID, testInfo.testScope);
											var testPlot = testInfo.testPlot + ".png";
											// descr_str = entry.descr + " " + testPlot;
											descr_str = "<a href='' onClick='Folder=\""+testFolder+"\";Scope=\""+testInfo.testScope+
												"\";Canvas=\""+testPlot+"\";return updateBrowserPage(\""+testPlot+"\",false,3)'>"+
												add_show_icon()+ entry.descr + "</a> ";
										}
									}
									//out += "<tr><td class='SEVERITY_"+sev_name+"'>" + sev_str+ "</td><td>"+ descr_str + "</td></tr>";
									out += "<tr><td class='SEVERITY' style='background-color:"+DQM_SEVERITY[sev_idx].hex+"'>" + sev_str+ "</td><td>"+ descr_str + "</td></tr>";
								}
							}
						}
						out += "</tr>";
						setEntryStatus(report[i].objID, obj_sev);
						// addlog(DQM_REPORT.report[i].objID, "info");
						// setEntryStatus(DQM_REPORT.report[i]);
					}
				}
			}
			out += "</table>";
		} else {
			out = "DQM Report file for Run:" + RunNumber +" not found";
		}
		obj.innerHTML=out;
		// DQMReport = out;
	}
}


function getDQMReport()
{
	var fullUrl = ResultsFolder+RunNumber+".plots/" + JSONReport;
	rep_div = document.getElementById("report_entry_div");
	if (rep_div) rep_div.innerHTML = "";
	
	var req=false;
	if (window.XMLHttpRequest) {
		req = new XMLHttpRequest();
	} else if (window.ActiveXObject) {
		req = new ActiveXObject((navigator.userAgent.toLowerCase().indexOf('msie 5') != -1) ? "Microsoft.XMLHTTP" : "Msxml2.XMLHTTP");
	}
	req.open("GET",fullUrl,true); // true= asynch, false=wait until loaded
	
	req.onreadystatechange = function() {
		if (req.readyState == 4) {
			if (req.status==200) {
				DQM_REPORT = false;
				var reply = req.responseText;
				// Need to strip malformed array string to satisfy IE7
				reply = reply.replace(/,]/g,"]");
				reply = reply.replace("var ","");
				window.eval(reply);
				showDQMReport("report_div");
			} else if (req.status==404) {
				DQM_REPORT = false;
				addlog("Can not load " + fullUrl, 'warn');
				// showTextReport("report_div");
				showDQMReport("report_div");
			}
		}
	}
	req.send(null);	
	return false;
}


function printReportLegend()
{
	var out = "<table class='dqm_report'><tr><td><b>Severity legend:</b> ";
	for (var i=0; i<DQM_SEVERITY.length; i++) {
		//out += " <span class='SEVERITY_"+DQM_SEVERITY[i].name+"'>" + DQM_SEVERITY[i].name + "</span> ";
		out += " <span class='SEVERITY' style='background-color:"+DQM_SEVERITY[i].hex+"'>" + DQM_SEVERITY[i].name + "</span> ";
	}
	out += "</td></tr></table>";
	return out; 
}


function addToTestsShowList(list, testID, testName, scope)
{
	var inList = false;
	if (testsShowLists) {
		var entry = new Array();
		entry[0]=testID;
		entry[1]=testName;
		entry[2]=scope;
		if (!testsShowLists[list]) testsShowLists[list] = new Array();
		sList = testsShowLists[list];
		for (var i=0; i<sList.length; i++) {
			if (sList[i][0] == testID) inList = true;
		}
		if (!inList) sList.push(entry);
	}
	if (list == "Custom") showSlideShowList(list);
}


function openReportInWindow()
{
	var fullUrl = ResultsFolder+RunNumber+".plots/" + TextReport;
	var title = "DQM Summary Report for Run: " + RunNumber;
	try {
		var generator=window.open("report","dqm_report","height=600,width=800,resizable=1,scrollbars=1");
		if (window.focus) {generator.focus()}
		generator.document.write('<html><head><title>'+title+'</title>');
		generator.document.write("<style>body {color: #000000; background-color: #9090FF;margin: 2px;font-family:Arial;	overflow: auto;}"+
		"span.winTitle {	margin: 0;font-size: 12px;font-weight: bold;text-decoration: none;color: #000000;}"+
		"fieldset {background-color: #ddd;	border-style: solid;border-width: 1px;border-color: black;margin: 2px;padding: 2px;}"+
		"legend {border-style: solid;border-width: 1px;background-color:  #C0C0FF;}</style>");
		generator.document.write('</head><body>');
		generator.document.write("<fieldset><legend><span class='winTitle'>"+title+"</span></legend>");
		generator.document.write("<pre>"+DQMReport+"</pre>");
		generator.document.write("</fieldset>");
		// generator.document.write('<p><a
		// href="javascript:self.close()">Close</a> the popup.</p>');
		generator.document.write('</body></html>');
		generator.document.close();
		reportWindow=generator;
	} 
	catch (exc){
	}
	
}


function findTestPlotInfo(id)
{
	var testPlot = "";
	// addlog(id, "debug");
	if (TESTS_MAP && TESTS_MAP.bindings.length) {
		for (var i = 0; i < TESTS_MAP.bindings.length; i++) {
			map_entry = TESTS_MAP.bindings[i];
			if (map_entry && map_entry.testID && map_entry.testPlot && (map_entry.testID == id)) { 
				// addlog("Found " + id, "debug");
				return map_entry;
			}
		}
	}
	return false;
}


function findHWId(id)
{
	//var m_list = CSCMAP;
	var m_list = [];
	if (m_list.length>0) {
		for (var i=0; i<m_list.length; i++) {
			if (m_list[i][2] == id ) return m_list[i][0]+"/"+m_list[i][1];
		}
	}
	return "";
}


function getFolderName(id, scope) {
	if (scope == "EMU") return "EMU";
	if (scope == "DDU" && (id.search("DDU_") >=0)) return id;
	if (scope == "CSC" ) return findHWId(id);
	return id;
}


function findReportEntry(report, id)
{
	for (var i = 0; i < report.length; i++) {
		if (report[i] && report[i].objID && report[i].objID  == id) return report[i];
	}
	return false;
}


// show id's Report Status in report_entry_div 
function setTestsStatus(id)
{
	//showTestsList (CANVASES_LIST, "tests_div");
	var out = "<table class='dqm_report'><tr><th class='repScope' colspan=3><b>Status for " + id + "</b></th></tr>";
	if (DQM_REPORT && DQM_REPORT.report.length) {
		var repentry = findReportEntry(DQM_REPORT.report, id);
		if (repentry) {
			if (repentry.list && repentry.list.length) {
				err_list = repentry.list;
				for (var j=0; j< err_list.length; j++) {
					var entry = err_list[j];
					if (entry) {
						// out += "<tr><td></td><td>" + entry.descr + "</td><td>"+entry.severity+"</td></tr>";
						var sev_idx = parseInt(entry.severity);
						var sev_name = DQM_SEVERITY[sev_idx].name;
						var sev_str = "";
						if ( sev_idx > 0) {
							sev_str = "<span>" + sev_name + "</span>";
						} 
						var descr_str = entry.descr;
						if (entry.testID != "") {
							var testInfo = findTestPlotInfo(entry.testID);
							if (testInfo) {
								var testFolder = getFolderName(repentry.objID, testInfo.testScope);// getFolder()
								var testPlot = testInfo.testPlot + ".png";
								var t_obj = document.getElementById(testPlot);
								if (t_obj) t_obj.style.background = DQM_SEVERITY[sev_idx].hex;
								// descr_str = entry.descr + " " +testPlot;
								descr_str = "<a href='' onClick='Folder=\""+testFolder+"\";Scope=\""+testInfo.testScope+"\";Canvas=\""+testPlot+
									"\";return updateBrowserPage(\""+testPlot+"\",false,3)'>"+add_show_icon() + entry.descr + "</a> ";
							}
						}
						//out += "<tr><td class='SEVERITY_"+sev_name+"'>" + sev_str+ "</td><td>"+ descr_str + "</td></tr>";
						out += "<tr><td class='SEVERITY' style='background-color:"+DQM_SEVERITY[sev_idx].hex+"'>" + sev_str+ "</td><td>"+ descr_str + "</td></tr>";
					}
				}
			}
		}
	}
	out += "</table>";
	updateBrowserPage(Canvas,false,3)
	rep_div = document.getElementById("report_entry_div");
	if (rep_div) rep_div.innerHTML = out;
}


function setEntryStatus(objname, severity)
{
	if (objname && (objname.search("ME") >= 0 || objname.search("MB") >= 0)) {
		m_obj = document.getElementById(objname);
		if (m_obj) {
			// addlog(objname + " " + severity, "debug");
			m_obj.style.background = DQM_SEVERITY[severity].hex;
		}
	}
}


function add_show_icon()
{
	return "<span class='show_icon'>[&diams;]</span> ";
}


// Load available systems list for selected Run
function getSystemsList(id)
{
	var fullUrl = ResultsFolder+RunNumber+".plots/" + MUList;
	var r_link=document.getElementById("RunLink");
	if (r_link) r_link.innerHTML = "Run Number: <a class='RLink' href='" + baseURL+RunNumber+".plots/'>" + RunNumber + "</a>";

	var req=false;
	if (window.XMLHttpRequest) {
		req = new XMLHttpRequest();
	}
	else if (window.ActiveXObject) {
		req = new ActiveXObject((navigator.userAgent.toLowerCase().indexOf('msie 5') != -1) ? "Microsoft.XMLHTTP" : "Msxml2.XMLHTTP");
	}
	req.open("GET",fullUrl,true); // true= asynch, false=wait until loaded

	req.onreadystatechange = function() {
		if (req.readyState == 4) {
			if (req.status==200) {
				var reply = req.responseText;
				// Need to strip malformed array string to satisfy IE7
				reply = reply.replace(/,]/g,"]");
				reply = reply.replace("var ","");
				window.eval(reply);

				updateMappings();
				//showHWTable();
				addlog("Run " + RunNumber + ": Found "+num_chambers+" CSCs, "+num_dt_chambers+" DTs", "info");
				//showHWTable();
				// addlog(reply, 'debug');
				// showDQMReport("report_div");
				getDQMReport();
				updateBrowserPage(id,false,2);
			} else if (req.status==404) {
			}
		}
	}
	req.send(null);
	return false;
}


function updateMappings()
{
	// DDUs.length = 0;
	c_list = MU_LIST;
	showCSCTable("csctable_div");
	showDTTable("dttable_div");
	num_chambers=0;
	num_dt_chambers=0;
	if (c_list.length > 0) {
		c_tree = c_list[0];
		for ( var j=1; j<c_tree.length; j++) {
			key = c_tree[j][0];
			entries = c_tree[j][1];
			if (entries.length >0) {
				for (k=0; k<entries.length; k++) {
					enableTableEntry(key, entries[k]);
				}
			}
		}
	}
	// showDQMReport("report_div");
	// selectFolder(Folder);
}


function enableTableEntry(entry_key, entry)
{
	obj_sel = selectedObjectList[4];
	if (entry_key.search("EMU") >= 0) {
		obj = document.getElementById("cscCommon");
		if (obj) {
			obj.style.background = 'lightgreen';
			val = obj.innerHTML;
			emuID = "csc_" + entry;
			emuLink="<a href='#' id='" + emuID+ "' onClick='Folder=\""+entry+"\"; selectCommonFolder(\""+entry+"\");setTestsStatus(\""+entry+"\");return showPlot();'>EMU Summary</a>";
			obj.innerHTML = emuLink;
			if (obj_sel && (obj_sel.id.search(emuID) >= 0 )) {
				Folder = entry;
				selectCommonFolder(entry);
				// setTestsStatus(entry);
			}
		}
	}
	else if (entry_key.search("CSC") >=0) {
		//cscName = findCSCFromMap(entry_key, entry);
		cscName = entry;
		obj = document.getElementById(cscName); 
		if (obj) {
			num_chambers++;
			
			obj.style.background = 'lightgreen';
			val = obj.innerHTML;
			//cscF = entry_key+"/"+entry;
			cscF = idToDir(cscName)
			cscID = "csc_" + cscName;
			//cscID = cscName;
			//csclink="<a href='#' id='" + cscID+ "' onClick='Folder=\""+cscF+"\"; selectChamberFolder(\""+entry_key+"\",\""+entry+"\"); "+
			//	"setTestsStatus(\""+cscName+"\");findLinkedDDU(\""+cscName+"\"); return showPlot();'>"+val+"</a>";
			csclink="<div id='" + cscID+ "' onClick='Folder=\""+cscF+"\"; Scope=\"CSC\"; selectChamberFolder(\""+entry_key+"\",\""+entry+"\"); "+
				"setTestsStatus(\""+cscName+"\");return showPlot();'>"+val+"</div>";
			obj.innerHTML	 = csclink;
			addToFoldersShowList("All CSCs", cscF, cscName);
			if (obj_sel && (obj_sel.id == cscID)) {
				Folder = cscF;
				selectChamberFolder(entry_key,entry);
				// setTestsStatus(cscName);
			}
			obj2 = document.getElementById("csc"+cscName); 
			if (obj2) {
				obj2.style.background = 'lightgreen';
			}
		}
	} 
	else if (entry_key.search("DT") >=0) {
		dtName = entry;
		obj = document.getElementById(dtName); 
		if (obj) {
			num_dt_chambers++;
			
			obj.style.background = 'lightgreen';
			val = obj.innerHTML;
			dtF = idToDir(dtName)
			dtID = "dt_" + dtName;
			dtlink="<div id='" + dtID+ "' onClick='Folder=\""+dtF+"\"; Scope=\"DT\"; selectChamberFolder(\""+entry_key+"\",\""+entry+"\"); "+
				"setTestsStatus(\""+dtName+"\");return showPlot();'>"+val+"</div>";
			obj.innerHTML	 = dtlink;
			addToFoldersShowList("All DTs", dtF, dtName);
			if (obj_sel && (obj_sel.id == dtID)) {
				Folder = dtF;
				selectChamberFolder(entry_key,entry);
				// setTestsStatus(cscName);
			}
			obj2 = document.getElementById("dt"+dtName); 
			if (obj2) {
				obj2.style.background = 'lightgreen';
			}
		}
	} 
}


function findCSCFromMap(crate, slot, m_list)
{
	//m_list = CSCMAP;
	m_list = [];
	if (m_list.length>0) {
		for (var i=0; i<m_list.length; i++) {
			if ( (m_list[i][0] == crate) && (m_list[i][1] == slot) )
				return m_list[i][2];
		}
	}
	return "";
}


function selectSystemFolder(id)
{
	isFolderValid=true;	
	hw_id = "hw_"+id;
	csc_id = "csc_"+id;
	selectObject(hw_id, false, 1);
	selectObject(csc_id, false, 4);
	// selFolder=id;
	// addlog(selFolder, "debug");
	return false; 
}


function selectFolder(folder)
{
	if (folder.search("crate") >=0) {
		crate = folder.substring(folder.indexOf("crate"),folder.indexOf("/"));
		slot = folder.substring(folder.indexOf("slot"),folder.length);		
		selectChamberFolder(crate, slot);
	}
}


function selectCommonFolder(id)
{
	isFolderValid=true;	
	hw_id = "hw_"+id;
	csc_id = "csc_"+id;
	selectObject(hw_id, false, 1);
	selectObject(csc_id, false, 4);
	// addlog(id, "debug");
	// showPlot();
	return false; 
}


function selectChamberFolder(sys, id)
{
	isFolderValid=true;	
	tid = "";
	if (sys=="CSC") tid = "csc_"+id;
	if (sys=="DT") tid = "dt_"+id;
	selectObject(tid, false, 4);
	// addlog(id, "debug");
	// showPlot();
	return false; 
}


function selectTest(id, b_select)
{
	isFolderValid=true;	
	selectObject(id, b_select, 3);
	// addlog(selected_list[idx].id, "debug");
	// showPlot();
	return false; 
}


function updateBrowserPage(id, f_select, idx)
{
	isFolderValid=true;		
	selectObject(id, f_select, idx);
	// addlog(selected_list[idx].id, "debug");
	var fIter_old = fIter;
	var Folder_old = Folder;
	if (Scope == "ALL") {
		fIter = "";
		Folder = "common";
	}
	showPlot();
	fIter = fIter_old;
	Folder = Folder_old;
	return false; 
}


function decImageHeight()
{	
	imgHeight = imgHeight-20;
	if (imgHeight<minImageHeight) imgHeight=minImageHeight;
	obj = document.getElementById("imageHeight");
	if (obj) obj.value = imgHeight;
	showPlot();
}


function incImageHeight()
{
	imgHeight = imgHeight+20;
	if (imgHeight>maxImageHeight) imgHeight = maxImageHeight;	
	obj = document.getElementById("imageHeight");
	if (obj) obj.value = imgHeight;
	
	showPlot();
}


function selectPlotMode()
{
	var obj = document.getElementById("openExternal");
	if (obj) {
		if (obj.checked) {
			fOpenExternal=true;
		} else {
			fOpenExternal=false;
		}
	}
}


function selectShowReferenceMode()
{
	var obj = document.getElementById("showReference");
	if (obj) {
		if (obj.checked) {
			fShowReference=true;
		} else {
			fShowReference=false;
		}
	}
}


function selectShowIter1Mode()
{
	var obj = document.getElementById("showIter1");
	if (obj) {
		if (obj.checked) {
			fShowIter1=true;
			fIter = 'iter1'
		} else {
			fShowIter1=false;
			fIter = 'iterN'
		}
		showPlot();
	}
}


function selectObject(id, f_deselect, idx) {
	var obj = get_element(id);
	if (obj) {
		if(!f_deselect) {
			var o_olditem = selectedObjectList[idx];
			selectedObjectList[idx] = obj;
			if (o_olditem) selectObject(o_olditem.id, true, idx);
		}
		obj.style.fontWeight = f_deselect ? 'normal' : 'bold';
		obj.style.color = f_deselect ? '#000000' : '#0000ff';
		obj.style.border = f_deselect ? 'none':'1px solid blue';
	} 
}


var timerID = 0;
var tStart  = null;

var showList = new Array();
var showIdx = 0;
var folderIdx = 0;
var foldersSList = "";
var updateDelay = 5;


function setSelectedList() {
	var selObj = document.getElementById("showlist_selection");
	if (selObj) {
		var selIdx = selObj.selectedIndex;
		selList = selObj.options[selIdx].value;
		showSlideShowList(selList);
		showList = testsShowLists[selList];
	}
}


function setSelectedFoldersList() {
	var selObj = document.getElementById("folders_selection");
	if (selObj) {
		var selIdx = selObj.selectedIndex;
		selList = selObj.options[selIdx].value;
		foldersSListIdx = selIdx;
		foldersSList = selList;
		// addlog(selList, "info");
		// folderList = foldersShowLists[selList];
	}
}


function showNextSlide()
{
	showIdx++;
	showSlide();
}


function showPrevSlide()
{
	if (showIdx==0) showIdx=showList.length;
	showIdx--;
	showSlide();
}


function showSlide()
{
	if (showIdx >= showList.length || showIdx<0) { showIdx = 0;}
	if (showList[showIdx]) { 
		Canvas = showList[showIdx][0];
		Scope = showList[showIdx][2];
	}
	obj = document.getElementById("currentTest");
	// addlog(Canvas, "debug");
	showPlot();
	selectTest(Canvas, false);
	selectFolder(Folder, false);
	// loadFrame(Canvas, false, 3);
	selectObject("ssid_"+showIdx, false, 5);
	if (obj) obj.innerHTML = Canvas;
}


function updateSlideShowTimer() {
	if(timerID) {
		clearTimeout(timerID);
		clockID  = 0;
	}
	if(!tStart) tStart = new Date();
	var tDate = new Date();
	var tDiff = tDate.getTime() - tStart.getTime();
	tDate.setTime(tDiff);
	showSlide();
	if (foldersSListIdx > 0) {
		if (loopFoldersFirst) {
			fList = foldersShowLists[foldersSList];
			if (folderIdx >= fList.length-1) {
				folderIdx = 0;
				showIdx++;
			} else {
				folderIdx++;
			}
			Folder =  fList[folderIdx][0];
		};
		// addlog(Folder,"debug");
	} else {
		showIdx++;
	}
	timerID = setTimeout("updateSlideShowTimer()", updateDelay*1000);
}


function startSlideShow() {
	tStart = new Date();
	obj = document.getElementById("updateInterval");
	if (obj) {
		val = parseInt(obj.value);
		if (!isNaN(val) && val <=30) updateDelay = val;
	}
	showSlideShowList();
	if (foldersShowLists[foldersSList][0]) {
		Folder = foldersShowLists[foldersSList][0][0];
		folderIdx = 0;
	}
	timerID  = setTimeout("updateSlideShowTimer()", updateDelay*1000);
}


function stopSlideShow() {
	if(timerID) {
		clearTimeout(timerID);
		timerID  = 0;
	}
	tStart = null;
}


function resetSlideShow() {
	tStart = null;
	showIdx=0;
	folderIdx = 0;
	// document.theTimer.theTime.value = "00:00";
}


function initSlideShowList()
{
	document.write('<label>Tests:</label>');
	document.write('<select id="showlist_selection" onChange="setSelectedList();">');
	for (var s_list in testsShowLists) {
		document.write("<option value='"+s_list+"'>"+s_list);
	}
	// document.write("<option value='m1'>Custom");
	document.write('</select>');
	setSelectedList();
	document.write('<label>Folders:</label>');
	document.write('<select id="folders_selection" onChange="setSelectedFoldersList();">');
	for (var s_list in foldersShowLists) {
		document.write("<option value='"+s_list+"'>"+s_list);
		
	}
	// document.write("<option value='m1'>Custom");
	document.write('</select>');
	setSelectedFoldersList();
}

function showSlideShowList(list)
{
	s_list = document.getElementById("slideshow_list");
	if (s_list && testsShowLists && testsShowLists[list]) {
		s_list.innerHTML = "";
		s = "";
		s += '<table width=100%>';
		if (list == "Custom") {
			s += "<tr><td><input type=button name=\"clear\" value=\"Clear\" onclick=\"clearShowList('Custom');\"></td></tr>";
		}
		sList = testsShowLists[list];
		ctrlStr = "";
		for (var i=0; i< sList.length; i++) {
			if (list == "Custom") {
				ctrlStr = "<a class='remove_from_list' href='#' onClick='removeFromTestsShowList(\"Custom\",\""+i+"\");'>[x]</a>"
			}
			s += '<tr><td id="ssid_'+i+'">'+ctrlStr+sList[i][1]+'</td></tr>';
		}
		s += '</table>';
		s_list.innerHTML = s;
		// select("ssid_0", false, 5);
	}
}

function removeFromTestsShowList(list, idx)
{
	if (testsShowLists && testsShowLists[list]) {
		testsShowLists[list].splice(idx,1);
		showSlideShowList(list);
	}
}


function addToFoldersShowList(list, folderID, folderName)
{
	var inList = false;
	if (foldersShowLists) {
		var entry = new Array();
		entry[0]=folderID;
		entry[1]=folderName;
		
		if (!foldersShowLists[list]) foldersShowLists[list] = new Array();
		sList = foldersShowLists[list];
		for (var i=0; i<sList.length; i++) {			
			if (sList[i][0] == folderID) inList = true;
		}
		if (!inList) sList.push(entry);
	}
	// if (list == "Custom") showSlideShowList(list);
}

function clearShowList(id, list)
{
	if (id == "Tests") {
		if (testsShowLists && testsShowLists[list]) { 
			testsShowLists[list].length = 0;
			showSlideShowList(list);
		}
	} else if (id == "Folders") {
		if (foldersShowLists && foldersShowLists[list]) { 
			foldersShowLists[list].length = 0;
			// showSlideShowList(list);
		}
	}
	
}


function setSelectedTab(obj)
{
	o_selected = get_element("selected");
	if (o_selected) {
		o_selected.id = '';
	}
	obj.id = "selected";
   if (obj.title == "csc_view") {
	div_obj = get_element("csctable_div");
	div_obj.style.display = "";
	div_obj = get_element("dttable_div");
	div_obj.style.display = "none";
	div_obj = get_element("common_div");
	div_obj.style.display = "none";
	div_obj = get_element("report_div");
	div_obj.style.display = "none";
	div_obj = get_element("csccounters_div");
	div_obj.style.display = "none";
	hideId("folder_DT")
	showId("folder_CSC")
   }
   if (obj.title == "dt_view") {
	div_obj = get_element("csctable_div");
	div_obj.style.display = "none";
	div_obj = get_element("dttable_div");
	div_obj.style.display = "";
	div_obj = get_element("common_div");
	div_obj.style.display = "none";
	div_obj = get_element("report_div");
	div_obj.style.display = "none";
	div_obj = get_element("csccounters_div");
	div_obj.style.display = "none";
	hideId("folder_CSC")
	showId("folder_DT")
   }
   else if (obj.title == "common_view") {
	div_obj = get_element("csctable_div");
	div_obj.style.display = "none";
	div_obj = get_element("dttable_div");
	div_obj.style.display = "none";
	div_obj = get_element("common_div");
	div_obj.style.display = "";
	div_obj = get_element("report_div");
	div_obj.style.display = "none";
	div_obj = get_element("csccounters_div");
	div_obj.style.display = "none";
   }
   else if (obj.title == "report_view") {
	div_obj = get_element("csctable_div");
	div_obj.style.display = "none";
	div_obj = get_element("dttable_div");
	div_obj.style.display = "none";
	div_obj = get_element("common_div");
	div_obj.style.display = "none";
	div_obj = get_element("report_div");
	div_obj.style.display = "";
	div_obj = get_element("csccounters_div");
	div_obj.style.display = "none";
   }
   else if (obj.title == "csccounters_view") {
	div_obj = get_element("csctable_div");
	div_obj.style.display = "none";
	div_obj = get_element("dttable_div");
	div_obj.style.display = "none";
	div_obj = get_element("common_div");
	div_obj.style.display = "none";
	div_obj = get_element("report_div");
	div_obj.style.display = "none";
	div_obj = get_element("csccounters_div");
	div_obj.style.display = "";
   }
   o_selected = obj;
   return false;
}


function handleArrowKeys(evt) {
	evt = (evt) ? evt : ((window.event) ? event : null);
	if (evt) {
		obj_sel = selectedObjectList[4];
		if (obj_sel) {
			sys = null;
			id = null;
			id0 = null;
			ch = null;
			if (obj_sel.id.substr(0,3)=="csc") {
				sys = "CSC";
				id = obj_sel.id.substr(4);
				ch = idToChamber(id);
				id0 = id.substr(0,id.lastIndexOf(ch));
			}
			if (obj_sel.id.substr(0,2)=="dt") {
				sys = "DT";
				id = obj_sel.id.substr(3);
				ch = idToChamber(id);
				id0 = id.substr(0,id.lastIndexOf(ch));
			}
			//addlog(sys+" "+id+" "+id0+" "+ch+" "+evt.keyCode,"debug");
			if (sys && id && id0 && ch) switch (evt.keyCode) {
			case 37: // left
				newid = id0 + pad0X(parseInt(stripLeadingZeroes(ch))-1);
				//addlog(newid+" "+parseInt(ch)+" "+(parseInt(ch)-1)+" "+pad0X(parseInt(ch)-1),"debug");
				newel = get_element(newid);
				if (newel) {
					Folder = idToDir(newid);
					selectChamberFolder(sys,newid);
					setTestsStatus(newid);
					showPlot();
				}
				break;
			//case 38: // up
				//break;
			case 39: // right
				newid = id0 + pad0X(parseInt(stripLeadingZeroes(ch))+1);
				newel = get_element(newid);
				if (newel) {
					Folder = idToDir(newid);
					selectChamberFolder(sys,newid);
					setTestsStatus(newid);
					showPlot();
				}
				//elem.style.left = (parseInt(left) + 5) + "px";
				break;
			//case 40: // down
				//break;
			}
		}
	}
}

document.onkeyup = handleArrowKeys;

get_element = document.all ?
	function (s_id) { return document.all[s_id] } :
	function (s_id) { return document.getElementById(s_id) };
