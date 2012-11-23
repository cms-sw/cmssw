import sys
import os
import os.path
import logging
import random
import json

import FWCore.ParameterSet.SequenceTypes as sqt
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Modules as mod
import FWCore.ParameterSet.Types as typ
import FWCore.ParameterSet.Mixins as mix

from Vispa.Plugins.ConfigEditor.ConfigDataAccessor import ConfigDataAccessor
from FWCore.GuiBrowsers.FileExportPlugin import FileExportPlugin


def elem(elemtype, innerHTML='', html_class='', **kwargs):
    if html_class: #since 'class' cannot be used as a kwarg
        kwargs['class'] = html_class
    args = ' '.join(['%s="%s"' % i for i in kwargs.items()])
    if args:
        return "<%s %s>%s</%s>\n" % (elemtype, args, innerHTML, elemtype)
    else:
        return "<%s>%s</%s>\n" % (elemtype, innerHTML, elemtype)

def get_jquery():
    jquery_file = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'jquery-1.6.2.min.js')
    if os.path.exists(jquery_file):
        return elem('script', open(jquery_file).read(), type='text/javascript')
    else:
        return elem('script', type='text/javascript', src=JQUERY)

JQUERY = "http://code.jquery.com/jquery-1.6.2.min.js"
LXR = "http://cmslxr.fnal.gov/lxr/ident?"
CVS = "http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/"
CMSLOGO = "http://cms.cern.ch/iCMS/imgs/icms/CMSheader_left.gif"

CSS_TEMPLATE =\
"""
body {border-width:0;margin:0}
#header {width:100%;height:10%;background-color:#000}
#mlist {position:absolute;width:32%;top:10%;left:0;height:85%;overflow:auto;padding-left:20px}
#mview {position:absolute;width:65%;height:85%;top:10%;left:35%;overflow:auto}
#footer {position:absolute;top:95%;width:100%;height:5%;background-color:#000}
#cmslogo {float:left;vertical-align:middle;margin:5px}
#head_filedir {float:left;vertical-align:middle;margin:5px}
#searcharea {float:right;vertical-align:middle;margin-top:3px;margin-right:5px;position:relative}
#head_dirname {color:#ccc;font-size:12}
#head_filename {color:#fff;font-weight:bold;font-size:16}
#searchhider {display:none;position:relative;top:-10px}
#searchhider a {color:#fff;text-decoration:none}
#searchresult {display:none;-moz-border-radius:10px;border-radius:10px;border:2px solid #999;background-color:#ccc;padding:5px;z-index:1000;position:relative;height:700%}
#searchscroll {overflow:scroll;height:100%;width:100%}
#searchresult div {margin:3px;padding:2px}
.searchfrom {color:#999;font-size:small}

#mview_header {font-weight:bold;font-size:16;background-color:#cc6;-moz-border-radius:10px;border-radius:10px;margin-top:3px;padding:5px}
#mview_header span {border-radius:5px;-moz-border-radius:5px;padding-left:5px;padding-top:2px;margin-top:2px;margin-left:5px}
#mview_subheader {background-color:#ffffcc;-moz-border-radius:10px;border-radius:10px;margin-top:3px;padding:5px}
#mview_pset {background-color:#ffffcc;-moz-border-radius:10px;border-radius:10px;margin-top:3px;padding:5px}
.mview_label {color:#999}
.mview_value {border-radius:5px;-moz-border-radius:5px;padding-left:5px;padding-top:2px;margin-top:2px;margin-left:5px;background-color:#cc6}
.mview_value a {text-decoration:none}
.mview_value span {display:inline-block}

#footer_crumbs {float:left;font-size:small;font-weight:bold;color:#fff}
#footer_crumbs span {color:#000}
#footer_about {float:right}
#footer_about a {text-decoration:none;font-weight:bold;color:#fff}

.csearch {background-color:#fff}

.seq_contain {position:relative;left:-20px}
.seq_toggle {position:absolute;left:5px;top:2px;width:15px;vertical-align:middle;text-align:center;font-weight:bold;font-size:20}
.seq_right {position:relative;left:20px;border-radius:5px;-moz-border-radius:5px;padding-left:5px;padding-top:3px;margin-top:3px;width:auto;display:inline-block}
.seq_label {font-weight:bold;border-radius:5px;-moz-border-radius:5px;padding:3px 1px;display:inline-block;border-width:1px;border-style:solid}
.seq_expand {display:none;padding-left:20px}

.module {position:relative;border-radius:5px;-moz-border-radius:5px;padding-left:3px;padding-top:1px;margin-top:2px;padding-right:3px;padding-bottom:1px;margin-right:2px;display:inline-block;border-width:1px;border-style:solid}

.path {background-color:#3cf;border-color:#3cf}
.endpath {background-color:#9cf;border-color:#9cf}
.sequence {background-color:#99f;border-color:#99f}
.edanalyzer {background-color:#f00;border-color:#f00}
.edproducer {background-color:#0f0;border-color:#0f0}
.edfilter {background-color:#ff0;border-color:#ff0}
.outputmodule {background-color:#f0f;border-color:#f0f}
.source {background-color:#f9c;border-color:#f9c}
.service {background-color:#f96;border-color:#f96}
.sources {background-color:#fcc;border-color:#fcc}
.services {background-color:#fc6;border-color:#fc6}
.essource {background-color:#9cc;border-color:#9cc}
.esproducer {background-color:#3c9;border-color:#3c9}
.esprefer {background-color:#096;border-color:#096}
.essources {background-color:#9fc;border-color:#9fc}
.esproducers {background-color:#3f9;border-color:#3f9}
.esprefers {background-color:#0c6;border-color:#0c6}
.unknown_type {background-color:#ccc;border-color:#ccc}

.used {border-radius:5px;-moz-border-radius:5px;padding:2px 5px;margin-top:2px;margin-right:5px;border-width:1px;border-style:solid}
.member {border-radius:5px;-moz-border-radius:5px;padding:2px 5px;margin-top:2px;margin-right:5px;border-width:1px;border-style:solid}

.clickable {cursor:pointer}
span.clickable:hover {border-color:#fff}
div.module.clickable:hover {border-color:#fff}
div.seq_label.clickable:hover {border-color:#fff}

.pset {border:2px solid #000;background-color:#ffffcc;font-size:small;border-collapse:collapse}
.pset td {border-width:1px;border-color:#ccc; border-style:solid none;margin:0;padding:2px 4px}
.pset_label {font-weight:bold}
.pset_type {color:#aaa}

.pset_vector {list-style-type:none;border:1px solid #000;padding-left:5px;margin:1px}
.pset_int {font-family:courier;color:blue}
.pset_double {font-family:courier;color:purple}
.pset_string {font-family:courier;color:brown}
.pset_bool {font-family:courier;color:#f0f}
.pset_inputtag {font-family:courier}
.pset_othertype {font-family:courier}
.pset_placehold {display:none;color:#aaa}

#about_dialog {position:absolute;width:60%;height:40%;top:30%;left:20%;border-radius:10px;-moz-border-radius:10px;color:#fff;background-color:#00f;border:5px solid #003;display:none;text-align:center}
"""

JS_TEMPLATE =\
"""
var modules = {};
var sequences = {};
var pset_keys = {};
var pset_values = {};
var crumbs = [];
var n_results = 0;
var last_search = "";

var cvsroot = "%(cvs)s";
var lxrroot = "%(lxr)s";
if (data.process.cmssw) {
    lxrroot = lxrroot + "v=" + data.process.cmssw + ";";
}

function parse_data() {
    function _pset_visitor(pset, context) {
        var keys = [];
        var values = [];
        for (var i in pset) {
            var item = pset[i];
            var name = context + item.label;
            if (item.type == 'PSet') {
                result = _pset_visitor(item.value, name+'.');
                keys.concat(result.keys);
                values.concat(result.values);
            } else if (item.type == 'VPSet') {
                for (var j in item.value) {
                    result = _pset_visitor(item.value[j], name+'.');
                    if (j==0) {
                        keys.concat(result.keys); //assume they all have equivalent structure
                    }
                    values.concat(result.values);
                }
            } else if (item.type == 'VInputTag') {
                keys.push(name);
                for (var j in item.value) {
                    values.push(item.value[j][0].toString()); //only modulename
                }
            } else if (item.type == 'InputTag') {
                keys.push(name);
                values.push(item.value[0].toString());
            } else if (item.list) {
                keys.push(name);
                for (var j in item.value) {
                    values.push(item.value[j].toString());
                }
            } else {
                keys.push(name);
                values.push(item.value.toString());
            }   
        }
        return {keys:keys, values:values};
    }
    function _path_visitor(path) {
        if (path.path) {
            sequences[path.label] = path;
            for (var child in path.path) {
                _path_visitor(path.path[child]);
            }
        } else {
            _module_visitor(path);
        }                 
    }
    function _module_visitor(module) {
        modules[module.label] = module;
        if (module.pset) {
            var result = _pset_visitor(module.pset, '');
            pset_keys[module.label] = result.keys;
            pset_values[module.label] = result.values;
        }
    }
    function _handle_flat(modlist) {
        for (var i in modlist) {
            _module_visitor(modlist[i]);
        }
    }
    _handle_flat(data.source);
    _handle_flat(data.services);
    _handle_flat(data.essources);
    _handle_flat(data.esproducers);
    _handle_flat(data.esprefers);
    for (var p in data.paths) {var path = data.paths[p]; _path_visitor(path);}
    for (var p in data.endpaths) {var path = data.paths[p]; _path_visitor(path);} 
}

function ensure_visible(name) {
    if (! $("#"+name).is(":visible")) {
        $("#"+name).parents(".seq_expand").each(function() {
            if (! $(this).is(":visible")) {
                var id = $(this).attr("id").slice(7);
                toggle_sequence(id);
            }
        });
    }
    $("#mlist").animate({"scrollTop": $("#"+name).position().top}, 1000);
}

function show_module(name) {
    var module = modules[name];
    if (module) {
        add_breadcrumb("module", name);
        ensure_visible(name);
        show_mview(module);
    }
}

function show_sequence(name) {
    var sequence = sequences[name];
    if (sequence) {
        add_breadcrumb("sequence", name);
        ensure_visible(name);
        show_mview(sequence);
    }
}

function show_mview(module) {
    function build_memberlist(memberof) {
        var new_html = "";
        for (var i in memberof) {
            var sequence = sequences[memberof[i]];
            if (sequence) {
                new_html += "<span class='clickable member "+sequence.type.toLowerCase()+"' onclick='show_sequence(\\""+sequence.label+"\\");'>"+sequence.label+"</span>";
            } else {
                new_html += "<span class='member unknown_type'>"+memberof[i]+"</span>";
            }
        }
        return new_html;
    }
    function build_uselist(uses) {
        var new_html = "";
        for (var i in uses) {
            var module = modules[uses[i]];
            if (module) {
                new_html += "<span class='clickable used "+module.type.toLowerCase()+"' onclick='show_module(\\""+module.label+"\\");'>"+module.label+"</span>";
            } else {
                new_html += "<span class='used unknown_type'>"+uses[i]+"</span>";
            }
        }
        return new_html;
    }
    function build_pset(pset, context, toplevel) {
        var typemap = {"string": "string", "double": "double", "int32": "int", "int64": "int", "uint32": "int", "uint64": "int", "bool": "bool"};
        if (! toplevel) {
            var new_html = "<span class='pset_placehold clickable' id='placehold_"+context+"' onclick='pset_toggle(\\""+context+"\\");'>("+pset.length.toString()+" hidden)</span><table id='content_"+context+"' class='pset'>";    
        } else {
            var new_html = "<table class='pset'>";
        }
        for (var i in pset) {
            var context2 = context + "_" + i.toString();
            var item = pset[i];
            if (item.untracked) {
                var itemtype = "cms.untracked."+item.type;
            } else {
                var itemtype = "cms."+item.type;
            }
            if (item.list) {
                itemtype += "[" + item.value.length.toString() + "]";
                new_html += "<tr><td class='pset_label'>"+item.label+"</td><td class='pset_type clickable'  onclick='pset_toggle(\\""+context2+"\\");'>"+itemtype+"</td><td>";
            } else if (item.type == 'PSet') {
                new_html += "<tr><td class='pset_label'>"+item.label+"</td><td class='pset_type clickable'  onclick='pset_toggle(\\""+context2+"\\");'>"+itemtype+"</td><td>";
            } else {
                new_html += "<tr><td class='pset_label'>"+item.label+"</td><td class='pset_type'>"+itemtype+"</td><td>";
            }
            
            if (item.type == 'PSet') {
                new_html += build_pset(item.value, context2);
            } else if (item.type == 'VPSet') {
                new_html += "<span class='pset_placehold clickable' id='placehold_"+context2+"' onclick='pset_toggle(\\""+context2+"\\");'>("+item.value.length.toString()+" hidden)</span><ul class='pset_vector' id='content_"+context2+"'>";
                for (var j in item.value) {
                    new_html += "<li>"+build_pset(item.value[j], context2+"_"+j.toString())+"</li>";
                }
                new_html += "</ul>";
            } else if (item.type == 'VInputTag') {
                new_html += "<span class='pset_placehold clickable' id='placehold_"+context2+"' onclick='pset_toggle(\\""+context2+"\\");'>("+item.value.length.toString()+" hidden)</span><ul class='pset_vector' id='content_"+context2+"'>";
                for (var j in item.value) {
                    var tag = item.value[j];
                    var link = build_uselist([tag[0]]);
                    if (tag[1] || tag[2]) {
                        new_html += "<li><span class='pset_inputtag'>"+link+":"+tag[1]+":"+tag[2]+"</span></li>";
                    } else {
                        new_html += "<li><span class='pset_inputtag'>"+link+"</span></li>";
                    }
                }
                new_html += "</ul>";
            } else if (item.type == 'InputTag') {
                var tag = item.value;
                var link = build_uselist([tag[0]]);
                if (tag[1] || tag[2]) {
                    new_html += "<span class='pset_inputtag'>"+link+":"+tag[1]+":"+tag[2]+"</span>";
                } else {
                    new_html += "<span class='pset_inputtag'>"+link+"</span>";
                }
            } else if (item.list) {
                new_html += "<span class='pset_placehold clickable' id='placehold_"+context2+"' onclick='pset_toggle(\\""+context2+"\\");'>("+item.value.length.toString()+" hidden)</span><ul class='pset_vector' id='content_"+context2+"'>";
                var cmstype = item.type.slice(1);
                if (typemap[cmstype]) {
                    var css = typemap[cmstype];
                } else {
                    var css = "othertype";
                }
                for (var j in item.value) {
                    new_html += "<li><span class='pset_"+css+"'>"+item.value[j]+"</span></li>";
                }
                new_html += "</ul>";                
            } else {
                var cmstype = item.type;
                if (typemap[cmstype]) {
                    var css = typemap[cmstype];
                } else {
                    var css = "othertype";
                }
                new_html += "<span class='pset_"+css+"'>"+item.value+"</span>";
            }
            new_html += "</td></tr>";
        }
        new_html += "</table>";       
        return new_html;
    }        
    var header = "<span class='used "+module.type.toLowerCase()+"'>"+module.label+"</span>";
    $("#mview_header").html(header);
    var table_html = "<tr><td class='mview_label'>Module Type:</td><td class='mview_value'>"+module.type+"</td></tr>";
    if (module.file && module.line) {
        table_html += "<tr><td class='mview_label'>Defined in:</td><td class='mview_value'><a href='"+cvsroot+module.file+"'>"+module.file+"</a>:"+module.line+"</td></tr>";
    } else if (module.file) {
        table_html += "<tr><td class='mview_label'>Defined in:</td><td class='mview_value'><a href='"+cvsroot+module.file+"'>"+module.file+"</a></td></tr>";
    }
    if (module.class) {
        table_html += "<tr><td class='mview_label'>Module Class:</td><td class='mview_value'><a href='"+lxrroot+"i="+module.class+"'>"+module.class+"</a></td></tr>";
    }
    if (module.uses) {
        table_html += "<tr><td class='mview_label'>Uses:</td><td class='mview_value'>"+build_uselist(module.uses)+"</td></tr>";
    }
    if (module.usedby) {
        table_html += "<tr><td class='mview_label'>Used by:</td><td class='mview_value'>"+build_uselist(module.usedby)+"</td></tr>";
    }
    if (module.memberof) {
        table_html += "<tr><td class='mview_label'>Member of:</td><td class='mview_value'>"+build_memberlist(module.memberof)+"</td></tr>";
    }
    $("#mview_table").html(table_html);
    if (module.pset) {
        $("#mview_pset").html(build_pset(module.pset, "pset", true));
        $("#mview_pset").find("tr,li").filter(":even").css({"background-color": "#ffa"});
        $("#mview_pset").find("tr,li").filter(":odd").css({"background-color": "#ffc"});
        $("#mview_pset").find('[id^="content_"]').each(function() {
            var id = $(this).attr("id").slice(8);
            if ($(this).children().size() > 5) {
                pset_toggle(id);
            }
        });
    } else {
        $("#mview_pset").html("");
    }
}

function pset_toggle(item) {
    if ($("#content_"+item).is(":visible")) {
        $("#content_"+item).hide("fast");
        $("#placehold_"+item).show("fast");
    } else {
        $("#content_"+item).show("fast");
        $("#placehold_"+item).hide("fast");
    }
}

function add_breadcrumb(itemtype, item) {
    if (crumbs.length > 0 && crumbs[crumbs.length-1][1] == item) {

    } else {
        if (crumbs.push([itemtype, item]) > 5) {
            crumbs = crumbs.slice(1);
        }
    }
    var new_html = "";
    for (var crumb in crumbs) {
        var c = crumbs[crumb];
        if (c[0] == "module") {
            var module = modules[c[1]];
            if (module) {
                var css = "used clickable "+module.type.toLowerCase();
	    } else {
                var css = "used unknown_type";
            }
            new_html += "&lt; <span class='"+css+"' onclick='show_module(\\"" + c[1] + "\\");'>" + c[1] + "</span>";
        } else if (c[0] == "sequence") {
            var sequence = sequences[c[1]];
            if (sequences) {
                var css = "used clickable "+sequence.type.toLowerCase();
            } else {
                var css = "used unknown_type";
            }
            new_html += "&lt; <span class='"+css+"' onclick='show_sequence(\\"" + c[1] + "\\");'>" + c[1] + "</span>";
        } else if (c[0] == "search") {
            new_html += "&lt; <span class='used clickable csearch' onclick='do_search(\\"" + c[1] + "\\");'>\\"" + c[1] + "\\"?</span>";
        }
    }
    $("#footer_crumbs").html(new_html);    
}

function toggle_sequence(sequence) {
    if ($("#expand_"+sequence).is(":visible")) {
        $("#expand_"+sequence).hide("fast");
        $("#toggle_"+sequence).text("+");
    } else {
        $("#expand_"+sequence).show("fast");
        $("#toggle_"+sequence).text("-");
    }
}

function build_mlist() {
    function build_path(path) {
        if (path.path) {
            var new_html = "<div class='seq_contain'><div class='seq_toggle clickable' id='toggle_"+path.label+"'>+</div><div class='seq_right "+path.type.toLowerCase()+"'><div id='"+path.label+"' class='seq_label "+path.type.toLowerCase()+"'>"+path.label+"</div><div class='seq_expand' id='expand_"+path.label+"'>";
            for (var child in path.path) {
                new_html += build_path(path.path[child]);
            }
            new_html += "</div></div></div>";
            return new_html;
        } else {
            return build_module(path);
        }
    }
    function build_module(module) {
        return "<div id='"+module.label+"' class='module "+module.type.toLowerCase()+"'>"+module.label+"</div>";   
    }
    var new_html = "";
    if (data.source) {
        new_html += build_path({"label":"Source", "type": "Sources", "path": data.source});
    }
    if (data.services) {
        new_html += build_path({"label":"Services", "type": "Services", "path": data.services});
    }
    if (data.paths) {
        for (var path in data.paths) {
            new_html += build_path(data.paths[path]);
        }
    }
    if (data.endpaths) {
        for (var path in data.endpaths) {
            new_html += build_path(data.endpaths[path]);
        }
    }
    if (data.essources) {
        new_html += build_path({"label":"ESSources", "type": "ESSources", "path": data.essources});
    }
    if (data.esproducers) {
        new_html += build_path({"label":"ESProducers", "type": "ESProducers", "path": data.esproducers});
    }
    if (data.esprefers) {
        new_html += build_path({"label":"ESPrefers", "type": "ESPrefers", "path": data.esprefers});
    }
    $("#mlist").html(new_html);
    $("#mlist").find(".seq_toggle").each(function() {
        $(this).click(function() {
            var id=$(this).attr("id").slice(7);
            toggle_sequence(id);
        });
    });
    $("#mlist").find(".seq_label").each(function() {
        $(this).click(function() {show_sequence($(this).attr("id"));});
        $(this).addClass("clickable");
    });
    $("#mlist").find(".module").each(function() {
        $(this).click(function() {show_module($(this).attr("id"));});
        $(this).addClass("clickable");
    });
}

function do_search(query) {
    if (! query) {return;}
    add_breadcrumb("search", query);
    var results = {modules:[], modclass:[], modfile:[], sequences:[], seqfile:[], keys:[], values:[]};
    var pattern = new RegExp(query, "gi");
    for (var i in modules) {
        if (i.search(pattern) != -1) {
            results.modules.push(i);
        }
        if (modules[i].class.search(pattern) != -1) {
            results.modclass.push(i);
        }
        if (modules[i].file) {
            if (modules[i].file.search(pattern) != -1) {
                results.modfile.push(i);
            }
        }
    }
    for (var i in sequences) {
        if (i.search(pattern) != -1) {
            results.sequences.push(i);
        }
        if (sequences[i].file) {
            if (sequences[i].file.search(pattern) != -1) {
                results.seqfile.push(i);
            }
        }
    }
    for (var i in pset_keys) {
        for (var j in pset_keys[i]) {
            if (pset_keys[i][j].search(pattern) != -1) {
                results.keys.push(i);
            }
        }
    }
    for (var i in pset_values) {
        for (var j in pset_values[i]) {
            if (pset_values[i][j].search(pattern) != -1) {
                results.values.push(i);
            }
        }
    }
    var new_html = '';
    function _module_div(name, extra) {
        var new_html = '';
        var module = modules[name];
        if (modules[name]) {
            var module = modules[name];
            var onclick = "show_module(\\""+module.label+"\\");toggle_search();";
        } else if (sequences[name]) {
            var module = sequences[name];
            var onclick = "show_sequence(\\""+module.label+"\\");toggle_search();";
        } else {
            return '';
        }
        var label = name.replace(pattern, "<b>$&</b>");
        new_html += "<div><span class='used clickable "+module.type.toLowerCase()+"' onclick='"+onclick+"'>"+label+"</span><span class='searchfrom'>("+extra+")</span></div>";
        return new_html;
    }
    var hitlist = [];
    var searchmap = {"module name":results.modules, "module class":results.modclass, "module file":results.modfile, "module pset key":results.keys, "module pset value":results.values, "sequence name":results.sequences, "sequence file":results.seqfiles};
    for (var i in searchmap) {
        for (var j in searchmap[i]) {
            if (hitlist.indexOf(searchmap[i][j]) == -1) {
                new_html += _module_div(searchmap[i][j], i);
                hitlist.push(searchmap[i][j]);
            }
        }
    }
    n_results = hitlist.length;
    last_search = query;
    $("#searchscroll").html(new_html);
    $("#searchinput").val("search");
    if ($("#searchhider").is(":hidden")) {$("#searchhider").show("fast");}
    toggle_search(true);
}

function toggle_search(force) {
    if ($("#searchresult").is(":hidden") || force==true) {
        $("#searchresult").show("slow");
        $("#searchhider a").html("Hide search results");
    } else {
        $("#searchresult").hide("slow");
        $("#searchhider a").html("Show results for <b>"+last_search+"</b> ("+n_results.toString()+")");
    }
}

function about() {
    $("#about_dialog").toggle("slow");
}

$(document).ready(function() {
    parse_data();
    build_mlist();
    $('#searchinput').focus(function() {if ($(this).val() == 'search') {$(this).val('');}});
    $('#searchsubmit').click(function(event) {event.preventDefault();do_search($('#searchinput').val());});
});
""" % dict(cvs=CVS, lxr=LXR)

PAGE_TEMPLATE =\
"""
<html>
<head>
<title>%(title)s</title>
<style type="text/css">
%(css)s
</style>
%(jquery)s
<script type="text/javascript">
var data = %(json)s;
%(js)s
</script>
</head>
<body>
<div id="header">
    <div id="cmslogo"><img src="%(cmslogo)s" alt="CMS Logo" height=48 width=48></img></div>
    <div id="head_filedir">
        <div id="head_dirname">%(dirname)s</div>
        <div id="head_filename">%(filename)s</div>
    </div>
    <div id="searcharea">
	<div>
            <form id="searchform">
                <input id="searchinput" type="text" value="search">
	        <input id="searchsubmit" type="submit" value="Search">
            </form>
        </div>
        <div id='searchhider'><a href='#' onclick='toggle_search();'></a></div>
        <div id='searchresult'>
            <div id='searchscroll'></div>
        </div>
    </div>
</div>
<div id="mlist"></div>
<div id="mview">
    <div id="mview_header"></div>
    <div id="mview_subheader">
        <table id="mview_table"></table>
    </div>
    <div id="mview_pset"></div>
</div>
<div id="footer">
    <div id="footer_crumbs"></div>
    <div id="footer_about"><a href='#' onclick='about();'>About</a></div>    
</div>
<div id="about_dialog" onclick='about();'>
    <div><h2>CMSSW configuration-to-html converter</h2></div>
    <div>Written by Gordon Ball (Imperial College)</div>
    <div>Uses CMSSW Config Editor and jQuery</div>
</div>
</body>
</html>
"""

    


class HTMLExport(FileExportPlugin):
    
    plugin_name = 'HTML Export'
    file_types = ('html', )

    def __init__(self):
        FileExportPlugin.__init__(self)

    def export(self, data, filename, filetype):
        with open(filename, 'w') as f:
            f.write(self.produce(data))

    def produce(self, data):
        return PAGE_TEMPLATE % dict(title=data._filename, jquery=get_jquery(),
                                    css=CSS_TEMPLATE, js=JS_TEMPLATE, dirname='.',
                                    filename=data._filename, json=self.data_to_json(data),
                                    cmslogo=CMSLOGO)
        
            
    def data_to_json(self, data):
        cmssw = None
        if 'CMSSW_BASE' in os.environ:
            cmssw = os.environ['CMSSW_BASE'].split('/')[-1]
        elif 'CMSSW_RELEASE_BASE' in os.environ:
            cmssw = os.environ['CMSSW_RELEASE_BASE'].split('/')[-1]
        result = {'process': {'name': data.process().name_() if data.process() else '(no process)', 'src': data._filename, 'cmssw':cmssw}}
        toplevel = data.children(data.topLevelObjects()[0]) if data.process() else data.topLevelObjects()
        for tlo in toplevel:
            children = data.children(tlo)
            label = tlo._label
            if label in ('source', 'services'):
                result[label] = [{'class':data.classname(child), 'pset':self.pset_to_json(child.parameters_()), 'type':data.type(child), 'label':data.classname(child)} for child in children]
            elif label in ('essources', 'esproducers', 'esprefers'):
                result[label] = [self.module_to_json(data, child) for child in children]
            elif label in ('paths', 'endpaths'):
                result[label] = [self.path_to_json(data, child) for child in children]
        return json.dumps(result, indent=4)

    def pset_to_json(self, pset):
        result = []
        for k, v in pset.items():
            typename = v.pythonTypeName().split('.')[-1]
            item = {'label': k, 'type': typename}
            if not v.isTracked():
                item['untracked'] = True 
            if typename == 'PSet':
                item['value'] = self.pset_to_json(v.parameters_())
            elif typename == 'VPSet':   
                item['value'] = [self.pset_to_json(vv.parameters_()) for vv in v]
                item['list'] = True
            elif typename == 'VInputTag':
                v_it = []
                for vv in v:
                   if type(vv) == cms.InputTag:
                     v_it.append(vv)
                   elif type(vv) == str:
                     v_it.append(cms.InputTag(vv))
                   else:
                      raise "Unsupported type in VInputTag", type(vv)
                item['value'] = [(vv.moduleLabel, vv.productInstanceLabel, vv.processName) for vv in v_it]
                item['list'] = True
            elif typename == 'InputTag':
                item['value'] = [v.moduleLabel, v.productInstanceLabel, v.processName]
            elif isinstance(v, mix._ValidatingListBase):
                item['value'] = [str(vv) for vv in v]
                item['list'] = True
            else:
                item['value'] = v.pythonValue()
            result += [item]
        return result

    def module_to_json(self, data, module):
        return {
            'label':data.label(module),
            'class':data.classname(module),
            'file':data.pypath(module),
            'line':data.lineNumber(module),
            #'package':data.pypackage(module),
            'pset':self.pset_to_json(module.parameters_()),
            'type':data.type(module),
            'uses':data.uses(module),
            'usedby':data.usedBy(module),
            'memberof':data.foundIn(module)
        }

    def path_to_json(self, data, path):
        children = data.children(path)
        if data.isContainer(path):
            json_children = [self.path_to_json(data, child) for child in children]
            return {'type':data.type(path), 'label':data.label(path), 
                    'path':json_children, 'memberof': data.foundIn(path),
                    'file': data.pypath(path), 'line': data.lineNumber(path)}
                    #'package': data.pypackage(path)}
        else:
            return self.module_to_json(data, path)
            
            
                            

class HTMLExportStatic(FileExportPlugin):
  options_types={}
  plugin_name='HTML Export (Static)'
  file_types=('html',)
  def __init__(self):
    FileExportPlugin.__init__(self)
  
  def produce(self,data):
    def elem(elemtype,innerHTML='',html_class='',**kwargs):
      if html_class:
        kwargs['class']=html_class
      return "<%s %s>%s</%s>\n" % (elemtype,' '.join(['%s="%s"'%(k,v) for k,v in kwargs.items()]),innerHTML,elemtype)
    def div(innerHTML='',html_class='',**kwargs):
      return elem('div',innerHTML,html_class,**kwargs)
    
    def htmlPSet(pset):
      def linkInputTag(tag):
        inputtag=''
        if isinstance(tag,typ.InputTag):
          inputtag = tag.pythonValue()
        else:
          inputtag = tag
        if len(str(tag))==0:
          inputtag = '""'
        return inputtag

      pset_items_html=''
      for k,v in pset.items():
        if isinstance(v,mix._ParameterTypeBase):
          if isinstance(v,mix._SimpleParameterTypeBase):
            item_class='other'
            if isinstance(v,typ.bool):
              item_class='bool'
            if isinstance(v,typ.double):
              item_class='double'
            if isinstance(v,typ.string):
              item_class='string'
            if isinstance(v,(typ.int32, typ.uint32, typ.int64, typ.uint64)):
              item_class='int'
            pset_items_html+=elem('tr',
              elem('td',k,'param-name')
             +elem('td',v.pythonTypeName(),'param-class')
             +elem('td',v.pythonValue(),'param-value-%s'%item_class),
             'pset-item'
            )
          if isinstance(v,typ.InputTag):
            pset_items_html+=elem('tr',
              elem('td',k,'param-name')
             +elem('td',v.pythonTypeName(),'param-class')
             +elem('td',linkInputTag(v),'param-value-inputtag'),
             'pset-item'
            )
          if isinstance(v,typ.PSet):
            pset_html = ''
            if len(v.parameters_())==0:
              pset_items_html+=elem('tr',
                elem('td',k,'param-name')
               +elem('td',v.pythonTypeName(),'param-class')
               +elem('td','(empty)','label'),
               'pset-item'
              )
            else:
              pset_items_html+=elem('tr',
                elem('td',k,'param-name')
               +elem('td',v.pythonTypeName(),'param-class')
               +elem('td',htmlPSet(v.parameters_())),
               'pset-item'
              )
          if isinstance(v,mix._ValidatingListBase):
            list_html = ''
            if len(v)==0:
              list_html = elem('li','(empty)','label')
            else:
              if isinstance(v,typ.VInputTag):
                for vv in v:
                  list_html += elem('li',linkInputTag(vv),'param-value-inputtag pset-list-item')
              elif isinstance(v,typ.VPSet):
                for vv in v:
                  list_html += elem('li',htmlPSet(vv.parameters_()),'pset-list-item')
              else:
                item_class='other'
                if isinstance(v,typ.vbool):
                  item_class='bool'
                if isinstance(v,typ.vdouble):
                  item_class='double'
                if isinstance(v,typ.vstring):
                  item_class='string'
                if isinstance(v,(typ.vint32,typ.vuint32,typ.vint64,typ.vuint64)):
                  item_class='int'
                for vv in v:
                  if len(str(vv))==0:
                    vv = "''"
                  list_html += elem('li',vv,'pset-list-item param-value-%s'%item_class)
            pset_items_html+=elem('tr',
              elem('td',k,'param-name')
             +elem('td','%s[%s]'%(v.pythonTypeName(),len(v)),'param-class')
             +elem('td',elem('ol',list_html,'pset-list')),
             'pset-item'
            )
              
            
      return elem('table',pset_items_html,'pset')
      
    def htmlModule(mod):
      mod_label_html = div(elem('a',data.label(mod),'title',name=data.label(mod)),'module_label '+data.type(mod),onClick='return toggleModuleVisible(\'%s\')'%('mod_%s'%(data.label(mod))))
      
      mod_table = elem('table',
        elem('tr',elem('td','Type','label')+elem('td',data.type(mod)))
       +elem('tr',elem('td','Class','label')+elem('td',data.classname(mod))),
        'module_table')
        
      mod_pset = htmlPSet(mod.parameters_())
      
      mod_content_html = div(mod_table+mod_pset,'module_area',id='mod_%s'%data.label(mod))
      return div(mod_label_html+mod_content_html,'module',id='module_'+data.label(mod))
      
    def htmlPathRecursive(p):
      children = data.children(p)
      if children:
        seq_name='Sequence'
        if isinstance(p,sqt.Path):
          seq_name='Path'
        if isinstance(p,sqt.EndPath):
          seq_name='EndPath'
        seq_label_html = div(seq_name+' '+elem('span',data.label(p),'title')+' '+elem('span','[%s children hidden]'%len(children),'hidden',id='seq_hidden_%s'%data.label(p)),'sequence_label',onClick='return toggleSequenceVisible(\'%s\')'%data.label(p),id='seq_label_%s'%data.label(p))
        seq_inner_content_html = ''.join([htmlPathRecursive(c) for c in children])
        seq_content_html = div(seq_inner_content_html,'sequence_area',id='seq_%s'%data.label(p))
        return div(seq_label_html+seq_content_html,'sequence')
      else:
        return htmlModule(p)
        
    toplevel={}
    
    
    
    filter_html = elem('span','Filter  '+
                        elem('input',type='text',width=50,onkeyup="return doFilter();",id='input-filter'),
                        'right label')
    
    header_html = div('Config File Visualisation'+filter_html,'header')
    
    if data.process():
      for tlo in data.children(data.topLevelObjects()[0]):
        children = data.children(tlo)
        if children:
          toplevel[tlo._label]=children    
      path_html=''
      if 'paths' in toplevel:
        for path in toplevel['paths']:
          path_html += div(htmlPathRecursive(path),'path')
    
      file_html = div(elem('span','Process:')
                   +elem('span',data.process().name_(),'title')
                   +elem('span',data._filename,'right'),
                'file')
      head_html = elem('head',elem('title',data.process().name_()))
    else:
      toplevel['sequences']=[]
      toplevel['paths']=[]
      toplevel['modules']=[]
      for tlo in data.topLevelObjects():
        if data.type(tlo)=='Sequence':
          toplevel['sequences']+=[tlo]
        if data.type(tlo)=='Path':
          toplevel['paths']+=[tlo]
        if data.type(tlo) in ('EDAnalyzer','EDFilter','EDProducer','OutputModule'):
          toplevel['modules']+=[tlo]
      
      path_html = ''
      sequence_html = ''
      module_html = ''
      for path in toplevel['paths']:
        path_html += div(htmlPathRecursive(path),'path')
      for sequence in toplevel['sequences']:
        sequence_html += htmlPathRecursive(sequence)
      for module in toplevel['modules']:
        module_html += htmlModule(module)
      file_html = div(elem('span',data._filename,'right'),'file')
      path_html += sequence_html
      path_html += module_html
      head_html = elem('head',elem('title',data._filename))
    footer_html = div('gordon.ball','footer')
    
    
    
    style_html = elem('style',
    """
    .title{font-weight:bold}
    .label{color:grey}
    .header{position:fixed;top:0px;left:0px;width:100%;background:#33cc00;font-weight:bold;font-size:120%}
    .footer{position:fixed;bottom:0px;left:0px;width:100%;background:#33cc00;text-align:right}
    .canvas{padding:40px 10px 40px 10px}
    .file{position:relative;background:#bbb;width:100%;padding-left:5px}
    .right{position:absolute;right:5px}
    .sequence{border:1px solid #aaa}
    .sequence:hover{border 1px solid #00ffff}
    .sequence_label{background:lightskyblue;padding-left:5px}
    .sequence_label:hover{background:#fff}
    .sequence_label_hidden{background:grey;padding-left:5px}
    .sequence_area{padding:5px 0px 5px 5px}
    .edproducer{border:1px solid red;background-image:url('edproducer.png');background-position:center left;background-repeat:no-repeat;padding:0px 0px 0px 40px}
    .edfilter{border:1px solid green;background-image:url('edfilter.png');background-position:center left;background-repeat:no-repeat;padding:0px 0px 0px 40px}
    .edanalyzer{border:1px solid blue;background-image:url('edanalyzer.png');background-position:center left;background-repeat:no-repeat;padding:0px 0px 0px 40px}
    .outputmodule{border:1px solid green;background-image:url('outputmodule.png');background-position:center left;background-repeat:no-repeat;padding:0px 0px 0px 40px}
    .module{}
    .module_label:hover{background:#ccc;position:relative}
    .module_area{display:none;padding:5px 0px 15px 15px;background:beige}
    .pset{border-spacing:10px 1px;border:1px solid black}
    .pset-item{}
    .pset-list{list-style-type:none;margin:0px;padding:2px 2px 2px 2px;border:1px solid grey}
    .pset-list-item{border-top:1px solid lightgrey;border-bottom:1px solid lightgrey}
    .param-name{font-weight:bold}
    .param-class{color:grey}
    .param-value-int{font-family:courier;color:blue}
    .param-value-double{font-family:courier;color:purple}
    .param-value-string{font-family:courier;color:brown}
    .param-value-bool{font-family:courier;color:#f0f}
    .param-value-inputtag{font-family:courier;color:red}
    .param-value-other{font-family:courier}
    .path{}
    .hidden{display:none}
    """,
    type='text/css')
    
    script_html = elem('script',
    """
    function toggleModuleVisible(id) {
      var elem = document.getElementById(id);
      if (elem.style.display=='block') {
        elem.style.display='none';
      } else {
        elem.style.display='block';      
      }
    }
    
    function toggleSequenceVisible(id) {
      var area_elem = document.getElementById('seq_'+id);
      var hidden_elem = document.getElementById('seq_hidden_'+id);
      var label_elem = document.getElementById('seq_label_'+id);
      if (area_elem.style.display=='none') {
        area_elem.style.display='block';      
        hidden_elem.style.display='none';
        label_elem.className = 'sequence_label';
      } else {
        area_elem.style.display='none';
        hidden_elem.style.display='block';
        label_elem.className = 'sequence_label_hidden';
      }
    }
    
    function doFilter() {
      var text = document.getElementById('input-filter').value;
      var regex = new RegExp(text);
      for (var i=0;i<document.all.length;i++) {
        if (document.all(i).id.substr(0,7)=="module_") {
          var elem = document.all(i);
          var elem_name = elem.id.substr(7);
          if (regex.test(elem_name)) {
            elem.style.display='block';
          } else {
            elem.style.display='none';
          }
        }
      }
    }
    """,
    type='text/javascript')
    
    body_html = elem('body',script_html+header_html+footer_html+div(file_html+path_html,'canvas'))
    
    return elem('html',head_html+style_html+body_html)
    
  def export(self,data,filename,filetype):
    #if not data.process():
    #  raise "HTMLExport requires a cms.Process object"
    
    html = self.produce(data)
    
    if filetype=='html':
      htmlfile = open(filename,'w')
      htmlfile.write(html)
      htmlfile.close()
