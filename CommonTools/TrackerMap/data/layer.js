var TrackerLayer = {} ;
TrackerLayer.thisFile = "layer.js" ;
var choice = "fed";	 
var choices = new Array();
choices[0]="fed";choices[1]="fec";choices[2]="lv"; choices[3]="hv"; choices[4]="plot";
var option;
TrackerLayer.init = function()
 {
  document.getElementById(choice).setAttribute("style","stroke: black; stroke-width: 1") ;
  chooseMap = TrackerLayer.chooseMap;
  showData = TrackerLayer.showData;
  
	}
 
    TrackerLayer.chooseMap = function (evt) { 
	var myPoly3 = evt.currentTarget;
	if (evt.type == "mouseover") {
	    if(myPoly3.getAttribute("mapAttribute")==choice){
		document.getElementById(choice).setAttribute("style","cursor:crosshair; stroke: black; stroke-width: 1") ;
		}
		else{
		myPoly3.setAttribute("style","cursor:crosshair;") ;
		document.getElementById(choice).setAttribute("style","stroke: black; stroke-width: 1") ;    
	    }
		}
	if (evt.type == "click") {
	    choice = myPoly3.getAttribute("mapAttribute");
        for( var i=0; i < 5;i++){
		    if(choices[i]!=choice){
			  opacity=1;
			  document.getElementById(choices[i]).setAttribute("style","fill-opacity: "+opacity) ;}
	          }
		document.getElementById(choice).setAttribute("style", "stroke: black; stroke-width: 1") ;
	   }
	if (evt.type == "mouseout") {
        if(myPoly3.getAttribute("mapAttribute")==choice){
		document.getElementById(choice).setAttribute("style","cursor:crosshair; stroke: black; stroke-width: 1") ;
		}
        else{opacity=1;
        document.getElementById(choice).setAttribute("style","stroke: black; stroke-width: 1") ;
	   }
	    }
	}

	
	
TrackerLayer.showData = function (evt) {
	
	var myPoly2;
	var myPoly = evt.currentTarget;
	  if (evt.type == "mouseover") {
		  
	    var myTracker = myPoly.getAttribute("POS");
        var myMessage = myPoly.getAttribute("MESSAGE");
        var separator = myTracker.indexOf("connected");
        separator = separator + 15;
		var fedchannel = myTracker.substring(separator);
        fedchannel = fedchannel.substring(0,fedchannel.indexOf("/"));
		var myTracker2 = "FED: "+fedchannel;
		//var fec
		var lvId = myPoly.getAttribute("lv");
		var hvId = myPoly.getAttribute("hv");
		var fecId = myPoly.getAttribute("fec");
		var lvRackCrate = Math.floor(lvId/100);
		var fecCrateSlot = Math.floor(fecId/100000);
		//alert(fecCrateSlot);
		var fecCrate = Math.floor(fecCrateSlot/100);
		var fecSlot = fecCrateSlot - fecCrate*100;
		myTracker2 = myTracker2 + "    FEC (C/S):  " + fecCrate + "/" + fecSlot; 
		
        var lvRack = Math.floor(lvRackCrate/10);
		
		var lvCrate = lvRackCrate - lvRack*10;
		var length = hvId.length;
		var hvChannel = hvId.substring(length -1,length);
		//alert(lvRack+" "+lvCrate +" "+hvChannel);
		myTracker2 = myTracker2 + "    LV  (R/C): " + lvRack + "/" + lvCrate + "    HV :  00" + hvChannel; 
		myTracker = myTracker.substring(0,separator - 15);
        var myTracker1 = "  value="+myPoly.getAttribute("value");
        myTracker1 = myTracker1+"  count="+myPoly.getAttribute("count");
        var textfield=document.getElementById('line1');
        textfield.firstChild.nodeValue=myTracker;
        textfield=document.getElementById('line3');
        textfield.firstChild.nodeValue=myTracker1;
        textfield=document.getElementById('line2');
        textfield.firstChild.nodeValue=myTracker2;
        textfield=document.getElementById('line4');
	    textfield.firstChild.nodeValue=myMessage;
        opacity=0.8;
        myPoly.setAttribute("style","cursor:crosshair; fill-opacity: "+opacity) ;
     
     
	 }
    
	
	if (evt.type == "click") {
		
	    var detid = myPoly.getAttribute("detid");
	    var id = myPoly.getAttribute("id");
	    var layer = Math.floor(id/100000);
	    var capvids = myPoly.getAttribute("capvids");
	    var comma = capvids.indexOf(',');
	    var apvaddr = parseInt(capvids.substring(1,comma));
	    var crate = Math.floor(apvaddr/1000000);
	 
		var apvaddr1 = capvids.substring(1,comma);
	 
		var rest = capvids.substring(comma+1);
	    comma = rest.indexOf(',');if(comma==-1)comma=rest.indexOf(')');
	    var apvaddr2 = rest.substring(0,comma);
	    rest = rest.substring(comma+1);
		var apvaddr3 = "";
	    if(rest.length>5){comma=rest.indexOf(')');apvaddr3 = rest.substring(0,comma);}
		var lvId = myPoly.getAttribute("lv");
		var hvId = myPoly.getAttribute("hv");
		var fecId = myPoly.getAttribute("fec");
		var lvRack = Math.floor(lvId/1000);
		var hvRack = Math.floor(hvId/10000);
		var fecCrate = Math.floor(fecId/10000000);
		fecId = 1000000*fecCrate + Math.floor(fecId);
		
		var base1 = Math.floor(apvaddr1.substring(0,apvaddr1.length-2)+"00");
		var base2 = Math.floor(apvaddr2.substring(0,apvaddr2.length-2)+"00");
		var base3 = Math.floor(apvaddr3.substring(0,apvaddr3.length-2)+"00");
		
		
		var modules = new Array();
        
		
		if(choice=="plot"){  
		
		myPoly.setAttribute("style"," stroke: black; stroke-width: 1") ; 
		alert("DQM PLOTS HERE, now detId = " + detid);
		
		}
		
		
		if(choice=="lv"){  
		  
		  if(parent.loaded!=lvRack || option==3 || option==4 || option==5 || option==6 || option==7 || option==8){parent.loaded=lvRack;parent.remotewin.location.href=parent.servername+parent.tmapname+"psurack"+lvRack+".xml";option=1;myPoly.setAttribute("style"," stroke: black; stroke-width: 1") ; }
		  else if(option==1){
		     
				myPoly2=parent.remotewin.document.getElementById(lvId);
		        myPoly2.setAttribute("style"," stroke: black; stroke-width: 1") ; 
		        var cmodid = myPoly2.getAttribute("cmodid");
	            var listinit = cmodid.indexOf('(')+1; 
                var modlist = cmodid.substring(listinit);
                modules=modlist.split(",");
	            for( var imod=0; imod < (modules.length-1);imod++){
                   document.getElementById(modules[imod]).setAttribute("style","stroke: black; stroke-width: 1") ;
			       }
			  option=2;
			 }
		  else{
		       option=1;
		  
			 myPoly2=parent.remotewin.document.getElementById(lvId);
		     myPoly2.setAttribute("style"," stroke: black; stroke-width: 1") ; 
		     var cmodid = myPoly2.getAttribute("cmodid");
	         var listinit = cmodid.indexOf('(')+1; 
             var modlist = cmodid.substring(listinit);
             modules=modlist.split(",");
             for( var imod=0; imod < (modules.length-1);imod++){
                   document.getElementById(modules[imod]).setAttribute("style","") ;
			      }			
		     myPoly.setAttribute("style"," stroke: black; stroke-width: 1") ;
			 }
		}
	    
		if(choice=="hv"){  
		   if(parent.loaded!=lvRack || option==1 || option==2 || option==5 || option==6 || option==7 || option==8){parent.loaded=lvRack;parent.remotewin.location.href=parent.servername+parent.tmapname+"HVrack"+hvRack+".xml";option=3;myPoly.setAttribute("style"," stroke: black; stroke-width: 1") ;}
		   else if(option==3){
			  myPoly2 =parent.remotewin.document.getElementById(hvId);
		      myPoly2.setAttribute("style"," stroke: black; stroke-width: 1") ; 
		      var cmodid = myPoly2.getAttribute("cmodid");
	          var listinit = cmodid.indexOf('(')+1; 
              var modlist = cmodid.substring(listinit);
              modules=modlist.split(",");
	          for( var imod=0; imod < (modules.length-1);imod++){
              document.getElementById(modules[imod]).setAttribute("style","stroke: black; stroke-width: 1") ;
			  }
			
			option=4;
		    }
		  else{
		     option=3;
		     
			 myPoly2=parent.remotewin.document.getElementById(hvId);
		     myPoly2.setAttribute("style"," stroke: black; stroke-width: 1") ; 
		     var cmodid = myPoly2.getAttribute("cmodid");
	         var listinit = cmodid.indexOf('(')+1; 
             var modlist = cmodid.substring(listinit);
             modules=modlist.split(",");
             for( var imod=0; imod < (modules.length-1);imod++){
                   document.getElementById(modules[imod]).setAttribute("style","") ;
			      }			
		     myPoly.setAttribute("style"," stroke: black; stroke-width: 1") ;
			
	       }
		}
		
		if(choice=="fec"){  
	     	if(fecCrate!=parent.loaded || option==1 || option==2 || option==3 || option==4 || option==7 || option==8){parent.loaded=fecCrate;parent.remotewin.location.href=parent.servername+parent.tmapname+"feccrate"+fecCrate+".xml";option=5;myPoly.setAttribute("style"," stroke: black; stroke-width: 1") ;}
		    else if(option==5){
		    myPoly2=parent.remotewin.document.getElementById(fecId);
		    myPoly2.setAttribute("style"," stroke: black; stroke-width: 1") ; 
			var cmodid = myPoly2.getAttribute("cmodid");
	        var listinit = cmodid.indexOf('(')+1; 
            var modlist = cmodid.substring(listinit);
            modules=modlist.split(",");
	        for( var imod=0; imod < (modules.length-1);imod++){
            document.getElementById(modules[imod]).setAttribute("style","stroke: black; stroke-width: 1") ;
			   }
			option=6;
			}
			else{
		    option=5;
		     
			 myPoly2=parent.remotewin.document.getElementById(fecId);
		     myPoly2.setAttribute("style"," stroke: black; stroke-width: 1") ; 
		     var cmodid = myPoly2.getAttribute("cmodid");
	         var listinit = cmodid.indexOf('(')+1; 
             var modlist = cmodid.substring(listinit);
             modules=modlist.split(",");
             for( var imod=0; imod < (modules.length-1);imod++){
                   document.getElementById(modules[imod]).setAttribute("style","") ;
			      }			
		     myPoly.setAttribute("style"," stroke: black; stroke-width: 1") ;
			
	       }
		}
		
		
		
	    if(choice=="fed"){
		
		  if(crate!=parent.loaded || option==3 || option==4 || option==5 || option==6 || option==1 || option==2){
		    parent.loaded=crate;
			parent.remotewin.location.href=parent.servername+parent.tmapname+"crate"+crate+".xml";
			myPoly.setAttribute("style","stroke: black; stroke-width: 1") ;
			option=7;
			}
       	    
	       
		  else if(option==7){
			
			if(parent.remotewin.document.getElementById(apvaddr1)!=null){
		      styledef=parent.remotewin.document.getElementById(apvaddr1).getAttribute("style");
		      if(styledef==null||styledef=="")parent.remotewin.document.getElementById(apvaddr1).setAttribute("style"," stroke: black; stroke-width: 1") ; 
		      else parent.remotewin.document.getElementById(apvaddr1).setAttribute("style","");
			  for(var k=0; k<96; k++){
			    if(parent.remotewin.document.getElementById(base1+k)!=null){
		          var modId = parent.remotewin.document.getElementById(base1+k).getAttribute("cmodid");
			      if(document.getElementById(modId)!=null)document.getElementById(modId).setAttribute("style","stroke: black; stroke-width: 1") ;
			      }
		         } 
			}
	     	
			if(apvaddr2!=""&&parent.remotewin.document.getElementById(apvaddr2)!=null) {
			   styledef=parent.remotewin.document.getElementById(apvaddr2).getAttribute("style");
			   if(styledef==null||styledef=="")parent.remotewin.document.getElementById(apvaddr2).setAttribute("style"," stroke: black; stroke-width: 1") ; 
			   else parent.remotewin.document.getElementById(apvaddr2).setAttribute("style",""); 
			  for(var k=0; k<96; k++){
			  if(parent.remotewin.document.getElementById(base2+k)!=null){
			  var modId = parent.remotewin.document.getElementById(base2+k).getAttribute("cmodid");
			  if(document.getElementById(modId)!=null)document.getElementById(modId).setAttribute("style","stroke: black; stroke-width: 1") ;
			     }
			    }
			 } 
			
			if(apvaddr3!=""&&parent.remotewin.document.getElementById(apvaddr3)!=null) {
			   styledef=parent.remotewin.document.getElementById(apvaddr3).getAttribute("style");
			   if(styledef==null||styledef=="")parent.remotewin.document.getElementById(apvaddr3).setAttribute("style"," stroke: black; stroke-width: 1") ; 
			   else parent.remotewin.document.getElementById(apvaddr3).setAttribute("style",""); 
			   for(var k=0; k<96; k++){
			    if(parent.remotewin.document.getElementById(base3+k)!=null){ 
			    var modId = parent.remotewin.document.getElementById(base3+k).getAttribute("cmodid");
			    if(document.getElementById(modId)!=null)document.getElementById(modId).setAttribute("style","stroke: black; stroke-width: 1") ;
			     }
			    }
		   	  }
			  option=8;
			}   
	        
			else{
			  option=7;
			  if(parent.remotewin.document.getElementById(apvaddr1)!=null){
		      styledef=parent.remotewin.document.getElementById(apvaddr1).getAttribute("style");
		      if(styledef==null||styledef=="")parent.remotewin.document.getElementById(apvaddr1).setAttribute("style"," stroke: black; stroke-width: 1") ; 
		      else parent.remotewin.document.getElementById(apvaddr1).setAttribute("style","");
			  for(var k=0; k<96; k++){
			    if(parent.remotewin.document.getElementById(base1+k)!=null){
		          var modId = parent.remotewin.document.getElementById(base1+k).getAttribute("cmodid");
			      if(document.getElementById(modId)!=null)document.getElementById(modId).setAttribute("style","") ;
			      }
		         } 
			  }
	     	
			  if(apvaddr2!=""&&parent.remotewin.document.getElementById(apvaddr2)!=null) {
			   styledef=parent.remotewin.document.getElementById(apvaddr2).getAttribute("style");
			   if(styledef==null||styledef=="")parent.remotewin.document.getElementById(apvaddr2).setAttribute("style"," stroke: black; stroke-width: 1") ; 
			   else parent.remotewin.document.getElementById(apvaddr2).setAttribute("style",""); 
			  for(var k=0; k<96; k++){
			  if(parent.remotewin.document.getElementById(base2+k)!=null){
			  var modId = parent.remotewin.document.getElementById(base2+k).getAttribute("cmodid");
			  if(document.getElementById(modId)!=null)document.getElementById(modId).setAttribute("style","") ;
			     }
			    }
			  } 
			
			   if(apvaddr3!=""&&parent.remotewin.document.getElementById(apvaddr3)!=null) {
			   styledef=parent.remotewin.document.getElementById(apvaddr3).getAttribute("style");
			   if(styledef==null||styledef=="")parent.remotewin.document.getElementById(apvaddr3).setAttribute("style"," stroke: black; stroke-width: 1") ; 
			   else parent.remotewin.document.getElementById(apvaddr3).setAttribute("style",""); 
			   for(var k=0; k<96; k++){
			    if(parent.remotewin.document.getElementById(base3+k)!=null){ 
			    var modId = parent.remotewin.document.getElementById(base3+k).getAttribute("cmodid");
			    if(document.getElementById(modId)!=null)document.getElementById(modId).setAttribute("style","") ;
			     }
			    }
		   	  }
	        myPoly.setAttribute("style","stroke: black; stroke-width: 1") ;
		   }
        parent.window.setip(parent.servername+parent.tmapname+"layer"+layer+".html#"+detid);
	    }
    }
	   
	   if (evt.type == "mouseout") {
        var myPoly = evt.currentTarget;
        opacity=1;
        myPoly.setAttribute("style","cursor:default; fill-opacity: "+opacity) ;
        }
  }

