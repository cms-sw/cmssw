	function insertAfter(parentnode, newElement, referenceElement){
		parentnode.insertBefore(newElement, referenceElement.nextSibling);
	}
	
	function alternate() {
		var i,t,row,table, tables;
    	tables = document.getElementsByTagName("table");
    	for (t = 0; t < tables.length; ++t) {
      		table = tables[t];
      		if ( table.className == 'doctable' ) {
      			for (i = 1; (row = table.getElementsByTagName("tr")[i]); ++i)
        			if ( row.className != 'colgroup' )
        				if (i % 2 == 1) { row.className='odd'; }
        				else { row.className='even'; }
        		}
        	}
		}

    	
    /* --- ARASH --- */	
    function init() {	
	    var out = "";
	    var href = "";
	    var aText = "";
	    var mainpage = false;
	    var directories = false;
	    var re = "";
	    var tabs = "";
	    var version = "";
	    var menu = document.getElementsByTagName("div");

    	for (var m = 0; m < menu.length; m++)
		{
    		var tabs = document.getElementsByTagName("div")[m];
			
		    if (tabs.className == "tabs")	// ------ FOR VERSION 1.5.4
		    {
		    	var ul = tabs.getElementsByTagName("ul")[0];
		    	//alert(ul.innerHTML);
		    	var li = ul.getElementsByTagName("li");
		    	var alpha = false;
				for( var i = 0; i < li.length; i++ )
				{
					ahref = li[i].getElementsByTagName("a")[0].getAttribute("href");
					aText = li[i].getElementsByTagName("a")[0].innerHTML;
		 			
					re = new RegExp("Files"); if(re.test(aText)) continue;
					re = new RegExp("Examples"); if(re.test(aText)) continue;
					re = new RegExp("Directories");
					if(re.test(aText))
					{
						//var version = "";
						var path = location.href.split(/\//g);
						var j = 0;
						while (j < path.length)
						{
							if (path[j].search(/CMSSW_/g) > -1) version = path[j];
							j++;
						}
						ahref = "http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/?pathrev="+version;
						aText = "<span>CVS Directory</span>";
					}
					re = new RegExp("Classes");
					if(re.test(aText))
					{
						ahref = "classes.html";	/*fix 13/05/08: annotated */
						// FIX
						var path = location.href.split(/\//g); 
						var j = path.length - 2;
						if (path[j] != "html") ahref = "../../classes.html";		/*fix 13/05/08: annotated */
						// FIX_END
					}
					re = new RegExp("Related.*Pages"); if(re.test(aText)) aText = "Package Documentation";
					re = new RegExp("Class.*List"); if(re.test(aText)) {/*continue;*/}		/*fix 14/05/08: alphabetical */
					re = new RegExp("Alphabetical.*List"); if(re.test(aText)) {alpha = true;}		/*fix 13/05/08: annotated */
					
					out = out+(i==0 || alpha?"":" | ");
					
					if (alpha) alpha = false;
					
					if (li[i].getAttribute("class") && li[i].getAttribute("class") == "current")
					{
						out = out+"<a href=\""+ahref+"\" class=\"qindexHL\">"+aText+"</a>";
						re = new RegExp("Main.*Page");
						mainpage=re.test(aText);
					}			
					else out = out+"<a href=\""+ahref+"\" class=\"qindex\">"+aText+"</a>";
					
				}
				//alert(out);
				if (m == 0)
				{
					out = out + " | <a class=\"qindex\" href=\"https://twiki.cern.ch/twiki/bin/view/CMS/WorkBook\"><span>WorkBook</span></a>";
					out = out + " | <a class=\"qindex\" href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuide\"><span>Offline&nbsp;Guide</span></a>";
				}
		    	tabs.setAttribute("class","qindex");
		    	tabs.innerHTML = out;
		    	out = "";
			}
			else if (tabs.className=="qindex" && m==0)	// ------ FOR VERSION 1.4.1
			{
				var namespaceHL = false;
				var clasHL = false;
				//if (location.href.search(/namespace_List/g) > -1)
				if (location.href.search(/namespaces/g) > -1)
				{
					namespaceHL = true;
				}
				//if (location.href.search(/namespace_Members/g) > -1)
				if (location.href.search(/namespacemembers/g) > -1)
				{
					namespaceHL = true;
				}
				//if (location.href.search(/class_hierarchy/g) > -1)
				if (location.href.search(/hierarchy/g) > -1)
				{
					clasHL = true;
				}
				//if (location.href.search(/alphabetical_list/g) > -1)
				if (location.href.search(/classes/g) > -1)		/*fix 13/05/08: annotated */
				{
					clasHL = true;
				}
				//if (location.href.search(/class_list/g) > -1)
				if (location.href.search(/annotated/g) > -1)	/*fix 14/05/08: alphabetical */
				{
					clasHL = true;
				}
				//if (location.href.search(/class_members/g) > -1)
				if (location.href.search(/functions/g) > -1)
				{
					clasHL = true;
				}

				out = "";
				var namespace = "";
				var nchk = false;
				var clas = "";
				var cchk = false;
				var a = tabs.innerHTML.split(/\s*\|\s*/g);
				var tmp = a[2];	/*fix 13/05/08: annotated */
				a[2] = a[3];
				a[3] = tmp;
				
				for( var i = 0; i < a.length; i++ )
				{
					if (i==1)
					{
						var namespacehref = " href=\"namespaces.html\">Namespaces</a>";
						var clashref = " href=\"classes.html\">Classes</a>";	/*fix 13/05/08: annotated */
						// FIX
						var path = location.href.split(/\//g); 
						var j = path.length - 2;
						if (path[j] != "html")
						{
							namespacehref = " href=\"../../namespaces.html\">Namespaces</a>";
							clashref = " href=\"../../classes.html\">Classes</a>";	/*fix 13/05/08: annotated */
						}
						// FIX_END
						out = out + " | <a class=\""+(namespaceHL?"qindexHL":"qindex")+"\""+namespacehref;
						out = out + " | <a class=\""+(clasHL?"qindexHL":"qindex")+"\""+clashref;
					}
					if (a[i].search(/File/g) > -1 || a[i].search(/Examples/g) > -1) continue;
					if (a[i].search(/Alphabetical/g) > -1)		/*fix 13/05/08: annotated */
					{
						clas = clas + (cchk?" | ":"") + a[i];
						cchk = true;
						continue;
					}
					if (a[i].search(/Namespace/g) > -1)
					{
						namespace = namespace + (nchk?" | ":"") + a[i];
						nchk = true;
						continue;
					}
					if (a[i].search(/Class/g) > -1)
					{
						/*fix 14/05/08: alphabetical */
						clas = clas + (cchk?" | ":"") + a[i];
						cchk = true;
						continue;
					}
					if (a[i].search(/Directories/g) > -1)
					{
						//var version = "";
						var path = location.href.split(/\//g);
						var j = 0;
						while (j < path.length)
						{
							if (path[j].search(/CMSSW_/g) > -1) version = path[j];
							j++;
						}
						var ahref = "http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/?pathrev="+version;
						//a[i] = a[i].replace(/Directories/g,"CVS Directory");
						a[i] = "<a class=\"qindex\" href=\""+ahref+"\">CVS Directory</a>";
					}
					if (a[i].search(/Related.*Pages/g) > -1) a[i] = a[i].replace(/Related.*Pages/g,"Package Documentation");
					
					out = out + (i==0?"":" | ") + a[i];
				}
				//alert(out);
					
				out = out + " | <a class=\"qindex\" href=\"https://twiki.cern.ch/twiki/bin/view/CMS/WorkBook\"><span>WorkBook</span></a>";
				out = out + " | <a class=\"qindex\" href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuide\"><span>Offline&nbsp;Guide</span></a>";
					
				tabs.innerHTML = out;
				if (namespaceHL)
				{
					var newDiv1 = document.createElement("div");
					newDiv1.setAttribute("class", "qindex");
					newDiv1.innerHTML = namespace;
					insertAfter(tabs.parentNode,newDiv1,tabs)
					//alert(newDiv1.innerHTML);
				}
				if (clasHL)
				{
					var newDiv2 = document.createElement("div");
					newDiv2.setAttribute("class", "qindex");
					newDiv2.innerHTML = clas;
					insertAfter(tabs.parentNode,newDiv2,tabs)
					//alert(newDiv2.innerHTML);				
				}
			}
    	}
  
 		var titleHTML = document.getElementsByTagName("title")[0];
 		var titletxt = document.createElement("div");
 		titleHTML.innerText = version + " Reference Manual";

    	
		if (location.href.search(/pages/g) > -1)
		{
			var pages = document.getElementsByTagName("h1")[0];
			pages.innerHTML = "CMSSW Package Documentation";
			pages.setAttribute("style","margin-bottom: 30px;");
			document.body.innerHTML = document.body.innerHTML.replace(/Here is a list of all related documentation pages:/g, "");
		}
		
		    
	    if (mainpage) alternate();
	}
	  
	  
	window.onload = init;
	
	
	
	
