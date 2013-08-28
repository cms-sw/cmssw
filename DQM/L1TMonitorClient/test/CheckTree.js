/*************************************************************
/*

CHECKTREE v1.0 RC (c) 2004-2006 Angus Turnbull, http://www.twinhelix.com
Altering this notice or redistributing this file is prohibited.

*/
/*************************************************************/

function CheckTree(myName){
  this.myName=myName;
  this.root=null;
  this.countAllLevels=false;
  this.checkFormat='(%n% checked)';
  this.evtProcessed=navigator.userAgent.indexOf('Safari')>-1?'safRtnVal':'returnValue';
//  alert("STEP0: CheckTree, name=" + myName);
  CheckTree.list[myName]=this
};

////////////////////////////////////////////////////////////////////////////////////////////////

CheckTree.list={};

CheckTree.prototype.init=function(){
 	debug_print("STEP2 inizio in Init");
 with(this){
  if(!parent.frames['status'].document.getElementById)return;
  root=parent.frames['status'].document.getElementById('tree-'+myName);
// 	debug_print("STEP2 CheckTree.prototype.init root= "+root);
  if(root){
       var lists=root.getElementsByTagName('ul');
       for(var ul=0;ul<lists.length;ul++){
          lists[ul].style.display='none';
	  lists[ul].treeObj=this;
	  lists[ul].setBoxStates=setBoxStates;
	  var fn=new Function('e','this.setBoxStates(e)');
	  if(lists[ul].addEventListener&&navigator.vendor!='Apple Computer,Inc.')
	    {
	    lists[ul].addEventListener('click',fn,false)
	    }
	    
	    else lists[ul].onclick=fn
	 }
 
 
	 root.treeObj=this;
	 root.setBoxStates=setBoxStates;
 	
		
	 if(root.addEventListener&&navigator.vendor!='Apple Computer,Inc.'){

	      root.addEventListener('click',new Function('e',myName+'.click(e)'),false)
 
	    } 
	    
	    else   root.onclick=new Function('e',myName+'.click(e)');

	    root.setBoxStates({},true,true);
	    var nodes=root.getElementsByTagName('li');
//	 debug_print("STEP2 CheckTree.prototype.init: nodes length = " + nodes.length);
	    for(var li=0;li<nodes.length;li++){
	       if(nodes[li].id.match(/^show-/)){
	           nodes[li].className=(nodes[li].className=='last'?'plus-last':'plus')
		}
	     }
	   }
	 }
 };
 
////////////////////////////////////////////////////////////////////////////////////////////////
 
 CheckTree.prototype.click=function(e){
// 	debug_print("CheckTree.prototype.click: called with this = "+ this);
 with(this){
// debug_print("CheckTree.prototype.click: event type = "+e.type);
 e=e||window.event;
 var elm=e.srcElement||e.target;
//    debug_print("CheckTree.prototype.click: element target of the event = "+elm);
 if(!e[evtProcessed]&&elm.id&&elm.id.match(/^check-(.*)/)){
    var tree=parent.frames['status'].document.getElementById('tree-'+RegExp.$1);
    if(tree)tree.setBoxStates(e,true,false)
    }
    while(elm){
//    debug_print("CheckTree.prototype.click: elm = "+elm);
    if(elm.tagName.match(/^(input|ul)/i))break;
    if(elm.id&&elm.id.match(/^show-(.*)/)){var targ=parent.frames['status'].document.getElementById('tree-'+RegExp.$1);
//    debug_print("CheckTree.prototype.click: targ = "+targ);
    if(targ.style){
      var col=(targ.style.display=='none');
      targ.style.display=col?'block':'none';
      elm.className=elm.className.replace(col?'plus':'minus',col?'minus':'plus')
      }
      break
      }elm=elm.parentNode }
    }
   };
   
////////////////////////////////////////////////////////////////////////////////////////////////   

   function setBoxStates(e,routingDown,countOnly){
   with(this){
   
   if(!this.childNodes)return;
   e=e||window.event;
   var elm=e.srcElement||e.target;
   if(elm&&elm.id&&elm.id.match(/^check-(.*)/)&&!routingDown&&!e[treeObj.evtProcessed])
     {
      var refTree=parent.frames['status'].document.getElementById('tree-'+RegExp.$1);
      if(refTree){refTree.setBoxStates(e,true,countOnly);
      e[treeObj.evtProcessed]=true
        }
      }
    
    var allChecked=true,boxCount=0,subBoxes=null;
    var thisLevel=this.id.match(/^tree-(.*)/)[1];
    var parBox=parent.frames['status'].document.getElementById('check-'+thisLevel);
    for(var li=0;li<childNodes.length;li++){
      for(var tag=0;tag<childNodes[li].childNodes.length;tag++){
         var child=childNodes[li].childNodes[tag];
	 if(!child)continue;
	 if(child.tagName&&child.type&&child.tagName.match(/^input/i)&&child.type.match(/^checkbox/i)){
	    if(routingDown&&parBox&&elm&&elm.id&&elm.id.match(/^check-/)&&!countOnly)child.checked=parBox.checked;
	    allChecked&=child.checked;
	    if(child.checked)boxCount++
	   }
	  if(child.tagName&&child.tagName.match(/^ul/i)&&(!e[treeObj.evtProcessed]||routingDown))child.setBoxStates(e,true,countOnly)
	  }
     }
     
     
     if(!routingDown)e[treeObj.evtProcessed]=true;
     if(parBox&&parBox!=elm&&!countOnly)parBox.checked=allChecked;
     if(treeObj.countAllLevels){
     boxCount=0;var subBoxes=this.getElementsByTagName('input');
     for(var i=0;i<subBoxes.length;i++)if(subBoxes[i].checked)boxCount++
     }
     
     var countElm=parent.frames['status'].document.getElementById('count-'+thisLevel);
     if(countElm){
       while(countElm.firstChild)countElm.removeChild(countElm.firstChild);
         if(boxCount)countElm.appendChild(parent.frames['status'].document.createTextNode(treeObj.checkFormat.replace('%n%',boxCount)))
	 }
	}
    };
    
//////////////////////////////////////////////////////////////////////////////      
// run this function when the page finishes loading so the field exists by the
// time this function is run. Problem is that the tree does not exist because
// it is created dynamically after  the page is loaded.
/*
    var chtOldOL=window.onload;window.onload=function(){
      if(chtOldOL)chtOldOL();
        for(var i in CheckTree.list){
 	// debug_print("STEP1:here onload;window.onload in loop");
          CheckTree.list[i].init()
        }
    };
*/

   function getTreeList(){
 	 debug_print("STEP1:in");
//      if(chtOldOL)chtOldOL();
        for(var i in CheckTree.list){
 	 debug_print("STEP1:here onload;window.onload in loop with i="+i);
          CheckTree.list[i].init()
        }
    }

