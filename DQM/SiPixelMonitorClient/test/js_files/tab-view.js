/************************************************************************************************************
(C) www.dhtmlgoodies.com, October 2005

This is a script from www.dhtmlgoodies.com. You will find this and a lot of other scripts at our website.

Terms of use:
You are free to use this script as long as the copyright message is kept intact. However, you may not
redistribute, sell or repost it without our permission.

Updated:

March, 14th, 2006 - Create new tabs dynamically
March, 15th, 2006 - Dynamically delete a tab

Thank you!

www.dhtmlgoodies.com
Alf Magne Kalleland

************************************************************************************************************/
   var textPadding = 3; // Padding at the left of tab text - bigger value gives you wider tabs
var strictDocType = true; 
var tabView_maxNumberOfTabs = 6;// Maximum number of tabs

/* Don't change anything below here */
var dhtmlgoodies_tabObj = new Array();
var activeTabIndex = new Array();
var MSIE = navigator.userAgent.indexOf('MSIE')>=0?true:false;
var navigatorVersion = navigator.appVersion.replace(/.*?MSIE (\d\.\d).*/g,'$1')/1;
var ajaxObjects = new Array();
var tabView_countTabs = new Array();
var tabViewHeight = new Array();
var tabDivCounter = 0;

function setPadding(obj,padding){
  var span = obj.getElementsByTagName('SPAN')[0];
  span.style.paddingLeft = padding + 'px';
  span.style.paddingRight = padding + 'px';
}
function showTab(parentId,tabIndex)
{
  var parentId_div = parentId + "_";
  if(!document.getElementById('tabView' + parentId_div + tabIndex)){
    return;
  }
  if(activeTabIndex[parentId]>=0){
    if(activeTabIndex[parentId]==tabIndex){
      return;
    }

    var obj = document.getElementById('tabTab'+parentId_div + activeTabIndex[parentId]);
    
    obj.className='tabInactive';
    var img = obj.getElementsByTagName('IMG')[0];
    img.src = '../images/tab_right_inactive.gif';
    document.getElementById('tabView' + parentId_div + activeTabIndex[parentId]).style.display='none';
  }
  
  var thisObj = document.getElementById('tabTab'+ parentId_div +tabIndex);
  
  thisObj.className='tabActive';
  var img = thisObj.getElementsByTagName('IMG')[0];
  img.src = '../images/tab_right_active.gif';
  
  document.getElementById('tabView' + parentId_div + tabIndex).style.display='block';
  activeTabIndex[parentId] = tabIndex;
  

  var parentObj = thisObj.parentNode;
  var aTab = parentObj.getElementsByTagName('DIV')[0];
  countObjects = 0;
  var startPos = 2;
  var previousObjectActive = false;
  while(aTab){
    if(aTab.tagName=='DIV'){
      if(previousObjectActive){
	previousObjectActive = false;
	startPos-=2;
      }
      if(aTab==thisObj){
	startPos-=2;
	previousObjectActive=true;
	setPadding(aTab,textPadding+1);
      }else{
	setPadding(aTab,textPadding);
      }
      
      aTab.style.left = startPos + 'px';
      countObjects++;
      startPos+=2;
    }
    aTab = aTab.nextSibling;
  }
  
  return;
}

function tabClick()
{
  var idArray = this.id.split('_');
  showTab(this.parentNode.parentNode.id,idArray[idArray.length-1].replace(/[^0-9]/gi,''));
  
}

function rolloverTab()
{
  if(this.className.indexOf('tabInactive')>=0){
    this.className='inactiveTabOver';
    var img = this.getElementsByTagName('IMG')[0];
    img.src = '../images/tab_right_over.gif';
  }
  
}
function rolloutTab()
{
  if(this.className ==  'inactiveTabOver'){
    this.className='tabInactive';
    var img = this.getElementsByTagName('IMG')[0];
    img.src = '../images/tab_right_inactive.gif';
  }
  
}

function initTabs(mainContainerID,tabTitles,activeTab,width,height,additionalTab)
{
  if(!additionalTab || additionalTab=='undefined'){
    dhtmlgoodies_tabObj[mainContainerID] = document.getElementById(mainContainerID);
    width = width + '';
    if(width.indexOf('%')<0)width= width + 'px';
    dhtmlgoodies_tabObj[mainContainerID].style.width = width;
    
    height = height + '';
    if(height.length>0){
      if(height.indexOf('%')<0)height= height + 'px';
      dhtmlgoodies_tabObj[mainContainerID].style.height = height;
    }
    

    tabViewHeight[mainContainerID] = height;
    
    var tabDiv = document.createElement('DIV');
    var firstDiv = dhtmlgoodies_tabObj[mainContainerID].getElementsByTagName('DIV')[0];
    
    dhtmlgoodies_tabObj[mainContainerID].insertBefore(tabDiv,firstDiv);
    tabDiv.className = 'dhtmlgoodies_tabPane';
    tabView_countTabs[mainContainerID] = 0;

  }else{
    var tabDiv = dhtmlgoodies_tabObj[mainContainerID].getElementsByTagName('DIV')[0];
    var firstDiv = dhtmlgoodies_tabObj[mainContainerID].getElementsByTagName('DIV')[1];
    height = tabViewHeight[mainContainerID];
    activeTab = tabView_countTabs[mainContainerID];

    
  }
  
  
  
  for(var no=0;no<tabTitles.length;no++){
    var aTab = document.createElement('DIV');
    aTab.id = 'tabTab' + mainContainerID + "_" +  (no + tabView_countTabs[mainContainerID]);
    aTab.onmouseover = rolloverTab;
    aTab.onmouseout = rolloutTab;
    aTab.onclick = tabClick;
    aTab.className='tabInactive';
    tabDiv.appendChild(aTab);
    var span = document.createElement('SPAN');
    span.innerHTML = tabTitles[no];
    aTab.appendChild(span);
    
    var img = document.createElement('IMG');
    img.valign = 'bottom';
    img.src = '../images/tab_right_inactive.gif';
    // IE5.X FIX
    if((navigatorVersion && navigatorVersion<6) || (MSIE && !strictDocType)){
      img.style.styleFloat = 'none';
      img.style.position = 'relative';
      img.style.top = '4px'
	span.style.paddingTop = '4px';
      aTab.style.cursor = 'hand';
    }// End IE5.x FIX
    aTab.appendChild(img);
  }

  var tabs = dhtmlgoodies_tabObj[mainContainerID].getElementsByTagName('DIV');
  var divCounter = 0;
  for(var no=0;no<tabs.length;no++){
    if(tabs[no].className=='dhtmlgoodies_aTab'){
      if(height.length>0)tabs[no].style.height = height;
      tabs[no].style.display='none';
      tabs[no].id = 'tabView' + mainContainerID + "_" + divCounter;
      divCounter++;
    }
  }
  tabView_countTabs[mainContainerID] = tabView_countTabs[mainContainerID] + tabTitles.length;
  showTab(mainContainerID,activeTab);

  return activeTab;
}

function showAjaxTabContent(ajaxIndex,parentId,tabId)
{
  var obj = document.getElementById('tabView'+parentId + '_' + tabId);
  obj.innerHTML = ajaxObjects[ajaxIndex].response;
}

function resetTabIds(parentId)
{
  var tabTitleCounter = 0;
  var tabContentCounter = 0;
  
  
  var divs = dhtmlgoodies_tabObj[parentId].getElementsByTagName('DIV');

  
  for(var no=0;no<divs.length;no++){
    if(divs[no].className=='dhtmlgoodies_aTab'){
      divs[no].id = 'tabView' + parentId + '_' + tabTitleCounter;
      tabTitleCounter++;
    }
    if(divs[no].id.indexOf('tabTab')>=0){
      divs[no].id = 'tabTab' + parentId + '_' + tabContentCounter;
      tabContentCounter++;
    }
    
    
  }

  tabView_countTabs[parentId] = tabContentCounter;
}


function createNewTab(parentId,tabTitle,tabContent,tabContentUrl)
{
  if(tabView_countTabs[parentId]>=tabView_maxNumberOfTabs)return;// Maximum number of tabs reached - return
  var div = document.createElement('DIV');
  div.className = 'dhtmlgoodies_aTab';
  dhtmlgoodies_tabObj[parentId].appendChild(div);
  var tabId = initTabs(parentId,Array(tabTitle),0,'','',true);
  if(tabContent)div.innerHTML = tabContent;
  if(tabContentUrl){
    var ajaxIndex = ajaxObjects.length;
    ajaxObjects[ajaxIndex] = new AjaxJs.sack();
    ajaxObjects[ajaxIndex].requestFile = tabContentUrl;// Specifying which file to get

    ajaxObjects[ajaxIndex].onCompletion = function(){ showAjaxTabContent(ajaxIndex,parentId,tabId); };// Specify function that will be executed after file has been found
    ajaxObjects[ajaxIndex].runAJAX();// Execute AJAX function
    
  }
  
}

function getTabIndexByTitle(tabTitle)
{
  for(var prop in dhtmlgoodies_tabObj){
    var divs = dhtmlgoodies_tabObj[prop].getElementsByTagName('DIV');
    for(var no=0;no<divs.length;no++){
      if(divs[no].id.indexOf('tabTab')>=0){
	var span = divs[no].getElementsByTagName('SPAN')[0];
	
	if(span.innerHTML == tabTitle){
	  var tmpId = divs[no].id.split('_');
	  
	  return Array(prop,tmpId[tmpId.length-1].replace(/[^0-9]/g,'')/1);
	}
      }
    }
  }
  
  return -1;
  
}


function deleteTab(tabLabel,tabIndex,parentId)
{
  
  if(tabLabel){
    var index = getTabIndexByTitle(tabLabel);
    if(index!=-1){
      deleteTab(false,index[1],index[0]);
    }
    
  }else if(tabIndex>=0){
    if(document.getElementById('tabTab' + parentId + '_' + tabIndex)){
      var obj = document.getElementById('tabTab' + parentId + '_' + tabIndex);
      var id = obj.parentNode.parentNode.id;
      obj.parentNode.removeChild(obj);
      var obj2 = document.getElementById('tabView' + parentId + '_' + tabIndex);
      obj2.parentNode.removeChild(obj2);
      resetTabIds(parentId);
      activeTabIndex[parentId]=-1;
      showTab(parentId,'0');
    }
  }
  

  
  
  
}
