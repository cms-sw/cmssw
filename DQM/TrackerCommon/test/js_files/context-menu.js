/************************************************************************************************************
(C) www.dhtmlgoodies.com, October 2005

This is a script from www.dhtmlgoodies.com. You will find this and a lot of other scripts at our website.	

Terms of use:
You are free to use this script as long as the copyright message is kept intact. However, you may not
redistribute, sell or repost it without our permission.

Thank you!

www.dhtmlgoodies.com
Alf Magne Kalleland

************************************************************************************************************/


var contextMenuObj;
var MSIE = navigator.userAgent.indexOf('MSIE')?true:false;
var navigatorVersion = navigator.appVersion.replace(/.*?MSIE (\d\.\d).*/g,'$1')/1;	
var activeContextMenuItem = false;
var contextMenuSource = false;	// Reference to element calling the context menu


document.documentElement.onclick = autoHideContextMenu;
function autoHideContextMenu(e)
{
  if(!contextMenuObj)return;
  if(document.all)e = event;
  if (e.target) source = e.target;
  else if (e.srcElement) source = e.srcElement;
  if (source.nodeType == 3) // defeat Safari bug
    source = source.parentNode;

  var tag1 = source;
  var tag2 = source;
  var tag3 = source;
  if(tag1.parentNode)tag2 = tag1.parentNode;
  if(tag1.parentNode.parentNode)tag3 = tag1.parentNode.parentNode;
  
  if(tag1.tagName!='contextMenu' && tag2.tagName!='contextMenu' && tag3.tagName!='contextMenu')contextMenuObj.style.display='none';	
	
}

function highlightContextMenuItem()
{
  this.className='contextMenuHighlighted';
}

function deHighlightContextMenuItem()
{
  this.className='';
}

function showContextMenu(e)
{
  contextMenuSource = this;
  if(activeContextMenuItem)activeContextMenuItem.className='';
  if(document.all)e = event;
  var xPos = e.clientX;
  if(xPos + contextMenuObj.offsetWidth > (document.documentElement.offsetWidth-20)){
    xPos = xPos + (document.documentElement.offsetWidth - (xPos + contextMenuObj.offsetWidth)) - 20;	
  }
	
  var yPos = e.clientY;
  if(yPos + contextMenuObj.offsetHeight > (document.documentElement.offsetHeight-20)){
    yPos = yPos + (document.documentElement.offsetHeight - (yPos + contextMenuObj.offsetHeight)) - 20;	
  }		
  contextMenuObj.style.left = xPos + 'px';
  contextMenuObj.style.top = yPos + 'px';
  contextMenuObj.style.display='block';
  return false;	
}

function hideContextMenu(e)
{
  if(document.all) e = event;
  if(e.button==0 && !MSIE){
    
  }else{
    contextMenuObj.style.display='none';
  }
}

function initContextMenu()
{
  contextMenuObj = document.getElementById('contextMenu');
  contextMenuObj.style.display = 'block';
  var menuItems = contextMenuObj.getElementsByTagName('LI');
  for(var no=0;no<menuItems.length;no++){
    menuItems[no].onmouseover = highlightContextMenuItem;
    menuItems[no].onmouseout = deHighlightContextMenuItem;
    
    var aTag = menuItems[no].getElementsByTagName('A')[0];
    
    var img = menuItems[no].getElementsByTagName('IMG')[0];
    if(img){
      var div = document.createElement('DIV');
      div.className = 'imageBox';
      div.appendChild(img);
      
      if(MSIE && navigatorVersion<6){
	aTag.style.paddingLeft = '0px';
      }
			
      var divTxt = document.createElement('DIV');	
      divTxt.className='itemTxt';
      divTxt.innerHTML = aTag.innerHTML;
			
      aTag.innerHTML = '';
      aTag.appendChild(div);
      aTag.appendChild(divTxt);
      if(MSIE && navigatorVersion<6){
	div.style.position = 'absolute';
	div.style.left = '2px';
	divTxt.style.paddingLeft = '15px';
      }
			
      if(!document.all){
	var clearDiv = document.createElement('DIV');
	clearDiv.style.clear = 'both';
	aTag.appendChild(clearDiv);		
      }
    }else{
      if(MSIE && navigatorVersion<6){
	aTag.style.paddingLeft = '15px';
	aTag.style.width = (aTag.offsetWidth - 30) + 'px';
      }else{
	aTag.style.paddingLeft = '30px';
	aTag.style.width = (aTag.offsetWidth - 60) + 'px';
      }
    }
  }
  contextMenuObj.style.display = 'none';		

}


	
