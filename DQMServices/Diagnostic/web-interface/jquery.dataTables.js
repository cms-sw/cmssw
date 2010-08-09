/*
 * File:        jquery.dataTables.js
 * Version:     1.6.2
 * CVS:         $Id$
 * Description: Paginate, search and sort HTML tables
 * Author:      Allan Jardine (www.sprymedia.co.uk)
 * Created:     28/3/2008
 * Modified:    $Date$ by $Author$
 * Language:    Javascript
 * License:     GPL v2 or BSD 3 point style
 * Project:     Mtaala
 * Contact:     allan.jardine@sprymedia.co.uk
 * 
 * Copyright 2008-2010 Allan Jardine, all rights reserved.
 *
 * This source file is free software, under either the GPL v2 license or a
 * BSD style license, as supplied with this software.
 * 
 * This source file is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
 * or FITNESS FOR A PARTICULAR PURPOSE. See the license files for details.
 * 
 * For details pleease refer to: http://www.datatables.net
 */

/*
 * When considering jsLint, we need to allow eval() as it it is used for reading cookies and 
 * building the dynamic multi-column sort functions.
 */
/*jslint evil: true, undef: true, browser: true */
/*globals $, jQuery,_fnReadCookie,_fnProcessingDisplay,_fnDraw,_fnSort,_fnReDraw,_fnDetectType,_fnSortingClasses,_fnSettingsFromNode,_fnBuildSearchArray,_fnCalculateEnd,_fnFeatureHtmlProcessing,_fnFeatureHtmlPaginate,_fnFeatureHtmlInfo,_fnUpdateInfo,_fnFeatureHtmlFilter,_fnFilter,_fnSaveState,_fnFilterColumn,_fnEscapeRegex,_fnFilterComplete,_fnFeatureHtmlLength,_fnGetDataMaster,_fnVisibleToColumnIndex,_fnDrawHead,_fnAddData,_fnGetTrNodes,_fnGetTdNodes,_fnColumnIndexToVisible,_fnCreateCookie,_fnAddOptionsHtml,_fnMap,_fnClearTable,_fnDataToSearch,_fnReOrderIndex,_fnFilterCustom,_fnNodeToDataIndex,_fnVisbleColumns,_fnAjaxUpdate,_fnAjaxUpdateDraw,_fnColumnOrdering,fnGetMaxLenString,_fnSortAttachListener,_fnPageChange*/

(function($) {
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * Section - DataTables variables
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	/*
	 * Variable: dataTableSettings
	 * Purpose:  Store the settings for each dataTables instance
	 * Scope:    jQuery.fn
	 */
	$.fn.dataTableSettings = [];
	var _aoSettings = $.fn.dataTableSettings; /* Short reference for fast internal lookup */
	
	/*
	 * Variable: dataTableExt
	 * Purpose:  Container for customisable parts of DataTables
	 * Scope:    jQuery.fn
	 */
	$.fn.dataTableExt = {};
	var _oExt = $.fn.dataTableExt;
	
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * Section - DataTables extensible objects
	 * 
	 * The _oExt object is used to provide an area where user dfined plugins can be 
	 * added to DataTables. The following properties of the object are used:
	 *   oApi - Plug-in API functions
	 *   aTypes - Auto-detection of types
	 *   oSort - Sorting functions used by DataTables (based on the type)
	 *   oPagination - Pagination functions for different input styles
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	/*
	 * Variable: sVersion
	 * Purpose:  Version string for plug-ins to check compatibility
	 * Scope:    jQuery.fn.dataTableExt
	 * Notes:    Allowed format is a.b.c.d.e where:
	 *   a:int, b:int, c:int, d:string(dev|beta), e:int. d and e are optional
	 */
	_oExt.sVersion = "1.6.2";
	
	/*
	 * Variable: iApiIndex
	 * Purpose:  Index for what 'this' index API functions should use
	 * Scope:    jQuery.fn.dataTableExt
	 */
	_oExt.iApiIndex = 0;
	
	/*
	 * Variable: oApi
	 * Purpose:  Container for plugin API functions
	 * Scope:    jQuery.fn.dataTableExt
	 */
	_oExt.oApi = { };
	
	/*
	 * Variable: aFiltering
	 * Purpose:  Container for plugin filtering functions
	 * Scope:    jQuery.fn.dataTableExt
	 */
	_oExt.afnFiltering = [ ];
	
	/*
	 * Variable: aoFeatures
	 * Purpose:  Container for plugin function functions
	 * Scope:    jQuery.fn.dataTableExt
	 * Notes:    Array of objects with the following parameters:
	 *   fnInit: Function for initialisation of Feature. Takes oSettings and returns node
	 *   cFeature: Character that will be matched in sDom - case sensitive
	 *   sFeature: Feature name - just for completeness :-)
	 */
	_oExt.aoFeatures = [ ];
	
	/*
	 * Variable: ofnSearch
	 * Purpose:  Container for custom filtering functions
	 * Scope:    jQuery.fn.dataTableExt
	 * Notes:    This is an object (the name should match the type) for custom filtering function,
	 *   which can be used for live DOM checking or formatted text filtering
	 */
	_oExt.ofnSearch = { };
	
	/*
	 * Variable: afnSortData
	 * Purpose:  Container for custom sorting data source functions
	 * Scope:    jQuery.fn.dataTableExt
	 * Notes:    Array (associative) of functions which is run prior to a column of this 
	 *   'SortDataType' being sorted upon.
	 *   Function input parameters:
	 *     object:oSettings-  DataTables settings object
	 *     int:iColumn - Target column number
	 *   Return value: Array of data which exactly matched the full data set size for the column to
	 *     be sorted upon
	 */
	_oExt.afnSortData = [ ];
	
	/*
	 * Variable: oStdClasses
	 * Purpose:  Storage for the various classes that DataTables uses
	 * Scope:    jQuery.fn.dataTableExt
	 */
	_oExt.oStdClasses = {
		/* Two buttons buttons */
		"sPagePrevEnabled": "paginate_enabled_previous",
		"sPagePrevDisabled": "paginate_disabled_previous",
		"sPageNextEnabled": "paginate_enabled_next",
		"sPageNextDisabled": "paginate_disabled_next",
		"sPageJUINext": "",
		"sPageJUIPrev": "",
		
		/* Full numbers paging buttons */
		"sPageButton": "paginate_button",
		"sPageButtonActive": "paginate_active",
		"sPageButtonStaticDisabled": "paginate_button",
		"sPageFirst": "first",
		"sPagePrevious": "previous",
		"sPageNext": "next",
		"sPageLast": "last",
		
		/* Stripping classes */
		"sStripOdd": "odd",
		"sStripEven": "even",
		
		/* Empty row */
		"sRowEmpty": "dataTables_empty",
		
		/* Features */
		"sWrapper": "dataTables_wrapper",
		"sFilter": "dataTables_filter",
		"sInfo": "dataTables_info",
		"sPaging": "dataTables_paginate paging_", /* Note that the type is postfixed */
		"sLength": "dataTables_length",
		"sProcessing": "dataTables_processing",
		
		/* Sorting */
		"sSortAsc": "sorting_asc",
		"sSortDesc": "sorting_desc",
		"sSortable": "sorting", /* Sortable in both directions */
		"sSortableAsc": "sorting_asc_disabled",
		"sSortableDesc": "sorting_desc_disabled",
		"sSortableNone": "sorting_disabled",
		"sSortColumn": "sorting_", /* Note that an int is postfixed for the sorting order */
		"sSortJUIAsc": "",
		"sSortJUIDesc": "",
		"sSortJUI": "",
		"sSortJUIAscAllowed": "",
		"sSortJUIDescAllowed": ""
	};
	
	/*
	 * Variable: oJUIClasses
	 * Purpose:  Storage for the various classes that DataTables uses - jQuery UI suitable
	 * Scope:    jQuery.fn.dataTableExt
	 */
	_oExt.oJUIClasses = {
		/* Two buttons buttons */
		"sPagePrevEnabled": "fg-button ui-state-default ui-corner-left",
		"sPagePrevDisabled": "fg-button ui-state-default ui-corner-left ui-state-disabled",
		"sPageNextEnabled": "fg-button ui-state-default ui-corner-right",
		"sPageNextDisabled": "fg-button ui-state-default ui-corner-right ui-state-disabled",
		"sPageJUINext": "ui-icon ui-icon-circle-arrow-e",
		"sPageJUIPrev": "ui-icon ui-icon-circle-arrow-w",
		
		/* Full numbers paging buttons */
		"sPageButton": "fg-button ui-state-default",
		"sPageButtonActive": "fg-button ui-state-default ui-state-disabled",
		"sPageButtonStaticDisabled": "fg-button ui-state-default ui-state-disabled",
		"sPageFirst": "first ui-corner-tl ui-corner-bl",
		"sPagePrevious": "previous",
		"sPageNext": "next",
		"sPageLast": "last ui-corner-tr ui-corner-br",
		
		/* Stripping classes */
		"sStripOdd": "odd",
		"sStripEven": "even",
		
		/* Empty row */
		"sRowEmpty": "dataTables_empty",
		
		/* Features */
		"sWrapper": "dataTables_wrapper",
		"sFilter": "dataTables_filter",
		"sInfo": "dataTables_info",
		"sPaging": "dataTables_paginate fg-buttonset fg-buttonset-multi paging_", /* Note that the type is postfixed */
		"sLength": "dataTables_length",
		"sProcessing": "dataTables_processing",
		
		/* Sorting */
		"sSortAsc": "ui-state-default",
		"sSortDesc": "ui-state-default",
		"sSortable": "ui-state-default",
		"sSortableAsc": "ui-state-default",
		"sSortableDesc": "ui-state-default",
		"sSortableNone": "ui-state-default",
		"sSortColumn": "sorting_", /* Note that an int is postfixed for the sorting order */
		"sSortJUIAsc": "css_right ui-icon ui-icon-triangle-1-n",
		"sSortJUIDesc": "css_right ui-icon ui-icon-triangle-1-s",
		"sSortJUI": "css_right ui-icon ui-icon-carat-2-n-s",
		"sSortJUIAscAllowed": "css_right ui-icon ui-icon-carat-1-n",
		"sSortJUIDescAllowed": "css_right ui-icon ui-icon-carat-1-s"
	};
	
	/*
	 * Variable: oPagination
	 * Purpose:  Container for the various type of pagination that dataTables supports
	 * Scope:    jQuery.fn.dataTableExt
	 */
	_oExt.oPagination = {
		/*
		 * Variable: two_button
		 * Purpose:  Standard two button (forward/back) pagination
	 	 * Scope:    jQuery.fn.dataTableExt.oPagination
		 */
		"two_button": {
			/*
			 * Function: oPagination.two_button.fnInit
			 * Purpose:  Initalise dom elements required for pagination with forward/back buttons only
			 * Returns:  -
	 		 * Inputs:   object:oSettings - dataTables settings object
	     *           node:nPaging - the DIV which contains this pagination control
			 *           function:fnCallbackDraw - draw function which must be called on update
			 */
			"fnInit": function ( oSettings, nPaging, fnCallbackDraw )
			{
				var nPrevious, nNext, nPreviousInner, nNextInner;
				
				/* Store the next and previous elements in the oSettings object as they can be very
				 * usful for automation - particularly testing
				 */
				if ( !oSettings.bJUI )
				{
					nPrevious = document.createElement( 'div' );
					nNext = document.createElement( 'div' );
				}
				else
				{
					nPrevious = document.createElement( 'a' );
					nNext = document.createElement( 'a' );
					
					nNextInner = document.createElement('span');
					nNextInner.className = oSettings.oClasses.sPageJUINext;
					nNext.appendChild( nNextInner );
					
					nPreviousInner = document.createElement('span');
					nPreviousInner.className = oSettings.oClasses.sPageJUIPrev;
					nPrevious.appendChild( nPreviousInner );
				}
				
				nPrevious.className = oSettings.oClasses.sPagePrevDisabled;
				nNext.className = oSettings.oClasses.sPageNextDisabled;
				
				nPrevious.title = oSettings.oLanguage.oPaginate.sPrevious;
				nNext.title = oSettings.oLanguage.oPaginate.sNext;
				
				nPaging.appendChild( nPrevious );
				nPaging.appendChild( nNext );
				
				$(nPrevious).click( function() {
					if ( oSettings.oApi._fnPageChange( oSettings, "previous" ) )
					{
						/* Only draw when the page has actually changed */
						fnCallbackDraw( oSettings );
					}
				} );
				
				$(nNext).click( function() {
					if ( oSettings.oApi._fnPageChange( oSettings, "next" ) )
					{
						fnCallbackDraw( oSettings );
					}
				} );
				
				/* Take the brutal approach to cancelling text selection */
				$(nPrevious).bind( 'selectstart', function () { return false; } );
				$(nNext).bind( 'selectstart', function () { return false; } );
				
				/* ID the first elements only */
				if ( oSettings.sTableId !== '' && typeof oSettings.aanFeatures.p == "undefined" )
				{
					nPaging.setAttribute( 'id', oSettings.sTableId+'_paginate' );
					nPrevious.setAttribute( 'id', oSettings.sTableId+'_previous' );
					nNext.setAttribute( 'id', oSettings.sTableId+'_next' );
				}
			},
			
			/*
			 * Function: oPagination.two_button.fnUpdate
			 * Purpose:  Update the two button pagination at the end of the draw
			 * Returns:  -
	 		 * Inputs:   object:oSettings - dataTables settings object
			 *           function:fnCallbackDraw - draw function to call on page change
			 */
			"fnUpdate": function ( oSettings, fnCallbackDraw )
			{
				if ( !oSettings.aanFeatures.p )
				{
					return;
				}
				
				/* Loop over each instance of the pager */
				var an = oSettings.aanFeatures.p;
				for ( var i=0, iLen=an.length ; i<iLen ; i++ )
				{
					if ( an[i].childNodes.length !== 0 )
					{
						an[i].childNodes[0].className = 
							( oSettings._iDisplayStart === 0 ) ? 
							oSettings.oClasses.sPagePrevDisabled : oSettings.oClasses.sPagePrevEnabled;
						
						an[i].childNodes[1].className = 
							( oSettings.fnDisplayEnd() == oSettings.fnRecordsDisplay() ) ? 
							oSettings.oClasses.sPageNextDisabled : oSettings.oClasses.sPageNextEnabled;
					}
				}
			}
		},
		
		
		/*
		 * Variable: iFullNumbersShowPages
		 * Purpose:  Change the number of pages which can be seen
	 	 * Scope:    jQuery.fn.dataTableExt.oPagination
		 */
		"iFullNumbersShowPages": 5,
		
		/*
		 * Variable: full_numbers
		 * Purpose:  Full numbers pagination
	 	 * Scope:    jQuery.fn.dataTableExt.oPagination
		 */
		"full_numbers": {
			/*
			 * Function: oPagination.full_numbers.fnInit
			 * Purpose:  Initalise dom elements required for pagination with a list of the pages
			 * Returns:  -
	 		 * Inputs:   object:oSettings - dataTables settings object
	     *           node:nPaging - the DIV which contains this pagination control
			 *           function:fnCallbackDraw - draw function which must be called on update
			 */
			"fnInit": function ( oSettings, nPaging, fnCallbackDraw )
			{
				var nFirst = document.createElement( 'span' );
				var nPrevious = document.createElement( 'span' );
				var nList = document.createElement( 'span' );
				var nNext = document.createElement( 'span' );
				var nLast = document.createElement( 'span' );
				
				nFirst.innerHTML = oSettings.oLanguage.oPaginate.sFirst;
				nPrevious.innerHTML = oSettings.oLanguage.oPaginate.sPrevious;
				nNext.innerHTML = oSettings.oLanguage.oPaginate.sNext;
				nLast.innerHTML = oSettings.oLanguage.oPaginate.sLast;
				
				var oClasses = oSettings.oClasses;
				nFirst.className = oClasses.sPageButton+" "+oClasses.sPageFirst;
				nPrevious.className = oClasses.sPageButton+" "+oClasses.sPagePrevious;
				nNext.className= oClasses.sPageButton+" "+oClasses.sPageNext;
				nLast.className = oClasses.sPageButton+" "+oClasses.sPageLast;
				
				nPaging.appendChild( nFirst );
				nPaging.appendChild( nPrevious );
				nPaging.appendChild( nList );
				nPaging.appendChild( nNext );
				nPaging.appendChild( nLast );
				
				$(nFirst).click( function () {
					if ( oSettings.oApi._fnPageChange( oSettings, "first" ) )
					{
						fnCallbackDraw( oSettings );
					}
				} );
				
				$(nPrevious).click( function() {
					if ( oSettings.oApi._fnPageChange( oSettings, "previous" ) )
					{
						fnCallbackDraw( oSettings );
					}
				} );
				
				$(nNext).click( function() {
					if ( oSettings.oApi._fnPageChange( oSettings, "next" ) )
					{
						fnCallbackDraw( oSettings );
					}
				} );
				
				$(nLast).click( function() {
					if ( oSettings.oApi._fnPageChange( oSettings, "last" ) )
					{
						fnCallbackDraw( oSettings );
					}
				} );
				
				/* Take the brutal approach to cancelling text selection */
				$('span', nPaging)
					.bind( 'mousedown', function () { return false; } )
					.bind( 'selectstart', function () { return false; } );
				
				/* ID the first elements only */
				if ( oSettings.sTableId !== '' && typeof oSettings.aanFeatures.p == "undefined" )
				{
					nPaging.setAttribute( 'id', oSettings.sTableId+'_paginate' );
					nFirst.setAttribute( 'id', oSettings.sTableId+'_first' );
					nPrevious.setAttribute( 'id', oSettings.sTableId+'_previous' );
					nNext.setAttribute( 'id', oSettings.sTableId+'_next' );
					nLast.setAttribute( 'id', oSettings.sTableId+'_last' );
				}
			},
			
			/*
			 * Function: oPagination.full_numbers.fnUpdate
			 * Purpose:  Update the list of page buttons shows
			 * Returns:  -
	 		 * Inputs:   object:oSettings - dataTables settings object
			 *           function:fnCallbackDraw - draw function to call on page change
			 */
			"fnUpdate": function ( oSettings, fnCallbackDraw )
			{
				if ( !oSettings.aanFeatures.p )
				{
					return;
				}
				
				var iPageCount = _oExt.oPagination.iFullNumbersShowPages;
				var iPageCountHalf = Math.floor(iPageCount / 2);
				var iPages = Math.ceil((oSettings.fnRecordsDisplay()) / oSettings._iDisplayLength);
				var iCurrentPage = Math.ceil(oSettings._iDisplayStart / oSettings._iDisplayLength) + 1;
				var sList = "";
				var iStartButton, iEndButton, i, iLen;
				var oClasses = oSettings.oClasses;
				
				/* Pages calculation */
				if (iPages < iPageCount)
				{
					iStartButton = 1;
					iEndButton = iPages;
				}
				else
				{
					if (iCurrentPage <= iPageCountHalf)
					{
						iStartButton = 1;
						iEndButton = iPageCount;
					}
					else
					{
						if (iCurrentPage >= (iPages - iPageCountHalf))
						{
							iStartButton = iPages - iPageCount + 1;
							iEndButton = iPages;
						}
						else
						{
							iStartButton = iCurrentPage - Math.ceil(iPageCount / 2) + 1;
							iEndButton = iStartButton + iPageCount - 1;
						}
					}
				}
				
				/* Build the dynamic list */
				for ( i=iStartButton ; i<=iEndButton ; i++ )
				{
					if ( iCurrentPage != i )
					{
						sList += '<span class="'+oClasses.sPageButton+'">'+i+'</span>';
					}
					else
					{
						sList += '<span class="'+oClasses.sPageButtonActive+'">'+i+'</span>';
					}
				}
				
				/* Loop over each instance of the pager */
				var an = oSettings.aanFeatures.p;
				var anButtons, anStatic, nPaginateList;
				var fnClick = function() {
					/* Use the information in the element to jump to the required page */
					var iTarget = (this.innerHTML * 1) - 1;
					oSettings._iDisplayStart = iTarget * oSettings._iDisplayLength;
					fnCallbackDraw( oSettings );
					return false;
				};
				var fnFalse = function () { return false; };
				
				for ( i=0, iLen=an.length ; i<iLen ; i++ )
				{
					if ( an[i].childNodes.length === 0 )
					{
						continue;
					}
					
					/* Build up the dynamic list forst - html and listeners */
					nPaginateList = an[i].childNodes[2];
					nPaginateList.innerHTML = sList;
					
					$('span', nPaginateList).click( fnClick ).bind( 'mousedown', fnFalse )
						.bind( 'selectstart', fnFalse );
					
					/* Update the 'premanent botton's classes */
					anButtons = an[i].getElementsByTagName('span');
					anStatic = [
						anButtons[0], anButtons[1], 
						anButtons[anButtons.length-2], anButtons[anButtons.length-1]
					];
					$(anStatic).removeClass( oClasses.sPageButton+" "+oClasses.sPageButtonActive+" "+oClasses.sPageButtonStaticDisabled );
					if ( iCurrentPage == 1 )
					{
						anStatic[0].className += " "+oClasses.sPageButtonStaticDisabled;
						anStatic[1].className += " "+oClasses.sPageButtonStaticDisabled;
					}
					else
					{
						anStatic[0].className += " "+oClasses.sPageButton;
						anStatic[1].className += " "+oClasses.sPageButton;
					}
					
					if ( iPages === 0 || iCurrentPage == iPages || oSettings._iDisplayLength == -1 )
					{
						anStatic[2].className += " "+oClasses.sPageButtonStaticDisabled;
						anStatic[3].className += " "+oClasses.sPageButtonStaticDisabled;
					}
					else
					{
						anStatic[2].className += " "+oClasses.sPageButton;
						anStatic[3].className += " "+oClasses.sPageButton;
					}
				}
			}
		}
	};
	
	/*
	 * Variable: oSort
	 * Purpose:  Wrapper for the sorting functions that can be used in DataTables
	 * Scope:    jQuery.fn.dataTableExt
	 * Notes:    The functions provided in this object are basically standard javascript sort
	 *   functions - they expect two inputs which they then compare and then return a priority
	 *   result. For each sort method added, two functions need to be defined, an ascending sort and
	 *   a descending sort.
	 */
	_oExt.oSort = {
		/*
		 * text sorting
		 */
		"string-asc": function ( a, b )
		{
			var x = a.toLowerCase();
			var y = b.toLowerCase();
			return ((x < y) ? -1 : ((x > y) ? 1 : 0));
		},
		
		"string-desc": function ( a, b )
		{
			var x = a.toLowerCase();
			var y = b.toLowerCase();
			return ((x < y) ? 1 : ((x > y) ? -1 : 0));
		},
		
		
		/*
		 * html sorting (ignore html tags)
		 */
		"html-asc": function ( a, b )
		{
			var x = a.replace( /<.*?>/g, "" ).toLowerCase();
			var y = b.replace( /<.*?>/g, "" ).toLowerCase();
			return ((x < y) ? -1 : ((x > y) ? 1 : 0));
		},
		
		"html-desc": function ( a, b )
		{
			var x = a.replace( /<.*?>/g, "" ).toLowerCase();
			var y = b.replace( /<.*?>/g, "" ).toLowerCase();
			return ((x < y) ? 1 : ((x > y) ? -1 : 0));
		},
		
		
		/*
		 * date sorting
		 */
		"date-asc": function ( a, b )
		{
			var x = Date.parse( a );
			var y = Date.parse( b );
			
			if ( isNaN( x ) )
			{
    		x = Date.parse( "01/01/1970 00:00:00" );
			}
			if ( isNaN( y ) )
			{
				y =	Date.parse( "01/01/1970 00:00:00" );
			}
			
			return x - y;
		},
		
		"date-desc": function ( a, b )
		{
			var x = Date.parse( a );
			var y = Date.parse( b );
			
			if ( isNaN( x ) )
			{
    		x = Date.parse( "01/01/1970 00:00:00" );
			}
			if ( isNaN( y ) )
			{
				y =	Date.parse( "01/01/1970 00:00:00" );
			}
			
			return y - x;
		},
		
		
		/*
		 * numerical sorting
		 */
		"numeric-asc": function ( a, b )
		{
			var x = a == "-" ? 0 : a;
			var y = b == "-" ? 0 : b;
			return x - y;
		},
		
		"numeric-desc": function ( a, b )
		{
			var x = a == "-" ? 0 : a;
			var y = b == "-" ? 0 : b;
			return y - x;
		}
	};
	
	
	/*
	 * Variable: aTypes
	 * Purpose:  Container for the various type of type detection that dataTables supports
	 * Scope:    jQuery.fn.dataTableExt
	 * Notes:    The functions in this array are expected to parse a string to see if it is a data
	 *   type that it recognises. If so then the function should return the name of the type (a
	 *   corresponding sort function should be defined!), if the type is not recognised then the
	 *   function should return null such that the parser and move on to check the next type.
	 *   Note that ordering is important in this array - the functions are processed linearly,
	 *   starting at index 0.
	 */
	_oExt.aTypes = [
		/*
		 * Function: -
		 * Purpose:  Check to see if a string is numeric
		 * Returns:  string:'numeric' or null
		 * Inputs:   string:sText - string to check
		 */
		function ( sData )
		{
			/* Sanity check that we are dealing with a string or quick return for a number */
			if ( typeof sData == 'number' )
			{
				return 'numeric';
			}
			else if ( typeof sData.charAt != 'function' )
			{
				return null;
			}
			
			var sValidFirstChars = "0123456789-";
			var sValidChars = "0123456789.";
			var Char;
			var bDecimal = false;
			
			/* Check for a valid first char (no period and allow negatives) */
			Char = sData.charAt(0); 
			if (sValidFirstChars.indexOf(Char) == -1) 
			{
				return null;
			}
			
			/* Check all the other characters are valid */
			for ( var i=1 ; i<sData.length ; i++ ) 
			{
				Char = sData.charAt(i); 
				if (sValidChars.indexOf(Char) == -1) 
				{
					return null;
				}
				
				/* Only allowed one decimal place... */
				if ( Char == "." )
				{
					if ( bDecimal )
					{
						return null;
					}
					bDecimal = true;
				}
			}
			
			return 'numeric';
		},
		
		/*
		 * Function: -
		 * Purpose:  Check to see if a string is actually a formatted date
		 * Returns:  string:'date' or null
		 * Inputs:   string:sText - string to check
		 */
		function ( sData )
		{
			var iParse = Date.parse(sData);
			if ( iParse !== null && !isNaN(iParse) )
			{
				return 'date';
			}
			return null;
		}
	];
	
	
	/*
	 * Variable: _oExternConfig
	 * Purpose:  Store information for DataTables to access globally about other instances
	 * Scope:    jQuery.fn.dataTableExt
	 */
	_oExt._oExternConfig = {
		/* int:iNextUnique - next unique number for an instance */
		"iNextUnique": 0
	};
	
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * Section - DataTables prototype
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	/*
	 * Function: dataTable
	 * Purpose:  DataTables information
	 * Returns:  -
	 * Inputs:   object:oInit - initalisation options for the table
	 */
	$.fn.dataTable = function( oInit )
	{
		/*
		 * Function: classSettings
		 * Purpose:  Settings container function for all 'class' properties which are required
		 *   by dataTables
		 * Returns:  -
		 * Inputs:   -
		 */
		function classSettings ()
		{
			this.fnRecordsTotal = function ()
			{
				if ( this.oFeatures.bServerSide ) {
					return this._iRecordsTotal;
				} else {
					return this.aiDisplayMaster.length;
				}
			};
			
			this.fnRecordsDisplay = function ()
			{
				if ( this.oFeatures.bServerSide ) {
					return this._iRecordsDisplay;
				} else {
					return this.aiDisplay.length;
				}
			};
			
			this.fnDisplayEnd = function ()
			{
				if ( this.oFeatures.bServerSide ) {
					return this._iDisplayStart + this.aiDisplay.length;
				} else {
					return this._iDisplayEnd;
				}
			};
			
			/*
			 * Variable: sInstance
			 * Purpose:  Unique idendifier for each instance of the DataTables object
			 * Scope:    jQuery.dataTable.classSettings 
			 */
			this.sInstance = null;
			
			/*
			 * Variable: oFeatures
			 * Purpose:  Indicate the enablement of key dataTable features
			 * Scope:    jQuery.dataTable.classSettings 
			 */
			this.oFeatures = {
				"bPaginate": true,
				"bLengthChange": true,
				"bFilter": true,
				"bSort": true,
				"bInfo": true,
				"bAutoWidth": true,
				"bProcessing": false,
				"bSortClasses": true,
				"bStateSave": false,
				"bServerSide": false
			};
			
			/*
			 * Variable: aanFeatures
			 * Purpose:  Array referencing the nodes which are used for the features
			 * Scope:    jQuery.dataTable.classSettings 
			 * Notes:    The parameters of this object match what is allowed by sDom - i.e.
			 *   'l' - Length changing
			 *   'f' - Filtering input
			 *   't' - The table!
			 *   'i' - Information
			 *   'p' - Pagination
			 *   'r' - pRocessing
			 */
			this.aanFeatures = [];
			
			/*
			 * Variable: oLanguage
			 * Purpose:  Store the language strings used by dataTables
			 * Scope:    jQuery.dataTable.classSettings
			 * Notes:    The words in the format _VAR_ are variables which are dynamically replaced
			 *   by javascript
			 */
			this.oLanguage = {
				"sProcessing": "Processing...",
				"sLengthMenu": "Show _MENU_ entries",
				"sZeroRecords": "No matching records found",
				"sInfo": "Showing _START_ to _END_ of _TOTAL_ entries",
				"sInfoEmpty": "Showing 0 to 0 of 0 entries",
				"sInfoFiltered": "(filtered from _MAX_ total entries)",
				"sInfoPostFix": "",
				"sSearch": "Search:",
				"sUrl": "",
				"oPaginate": {
					"sFirst":    "First",
					"sPrevious": "Previous",
					"sNext":     "Next",
					"sLast":     "Last"
				}
			};
			
			/*
			 * Variable: aoData
			 * Purpose:  Store data information
			 * Scope:    jQuery.dataTable.classSettings 
			 * Notes:    This is an array of objects with the following parameters:
			 *   int: _iId - internal id for tracking
			 *   array: _aData - internal data - used for sorting / filtering etc
			 *   node: nTr - display node
			 *   array node: _anHidden - hidden TD nodes
			 *   string: _sRowStripe
			 */
			this.aoData = [];
			
			/*
			 * Variable: aiDisplay
			 * Purpose:  Array of indexes which are in the current display (after filtering etc)
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.aiDisplay = [];
			
			/*
			 * Variable: aiDisplayMaster
			 * Purpose:  Array of indexes for display - no filtering
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.aiDisplayMaster = [];
							
			/*
			 * Variable: aoColumns
			 * Purpose:  Store information about each column that is in use
			 * Scope:    jQuery.dataTable.classSettings 
			 */
			this.aoColumns = [];
			
			/*
			 * Variable: iNextId
			 * Purpose:  Store the next unique id to be used for a new row
			 * Scope:    jQuery.dataTable.classSettings 
			 */
			this.iNextId = 0;
			
			/*
			 * Variable: asDataSearch
			 * Purpose:  Search data array for regular expression searching
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.asDataSearch = [];
			
			/*
			 * Variable: oPreviousSearch
			 * Purpose:  Store the previous search incase we want to force a re-search
			 *   or compare the old search to a new one
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.oPreviousSearch = {
				"sSearch": "",
				"bEscapeRegex": true
			};
			
			/*
			 * Variable: aoPreSearchCols
			 * Purpose:  Store the previous search for each column
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.aoPreSearchCols = [];
			
			/*
			 * Variable: aaSorting
			 * Purpose:  Sorting information
			 * Scope:    jQuery.dataTable.classSettings
			 * Notes:    Index 0 - column number
			 *           Index 1 - current sorting direction
			 *           Index 2 - index of asSorting for this column
			 */
			this.aaSorting = [ [0, 'asc', 0] ];
			
			/*
			 * Variable: aaSortingFixed
			 * Purpose:  Sorting information that is always applied
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.aaSortingFixed = null;
			
			/*
			 * Variable: asStripClasses
			 * Purpose:  Classes to use for the striping of a table
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.asStripClasses = [];
			
			/*
			 * Variable: fnRowCallback
			 * Purpose:  Call this function every time a row is inserted (draw)
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.fnRowCallback = null;
			
			/*
			 * Variable: fnHeaderCallback
			 * Purpose:  Callback function for the header on each draw
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.fnHeaderCallback = null;
			
			/*
			 * Variable: fnFooterCallback
			 * Purpose:  Callback function for the footer on each draw
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.fnFooterCallback = null;
			
			/*
			 * Variable: aoDrawCallback
			 * Purpose:  Array of callback functions for draw callback functions
			 * Scope:    jQuery.dataTable.classSettings
			 * Notes:    Each array element is an object with the following parameters:
			 *   function:fn - function to call
			 *   string:sName - name callback (feature). useful for arranging array
			 */
			this.aoDrawCallback = [];
			
			/*
			 * Variable: fnInitComplete
			 * Purpose:  Callback function for when the table has been initalised
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.fnInitComplete = null;
			
			/*
			 * Variable: sTableId
			 * Purpose:  Cache the table ID for quick access
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.sTableId = "";
			
			/*
			 * Variable: nTable
			 * Purpose:  Cache the table node for quick access
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.nTable = null;
			
			/*
			 * Variable: iDefaultSortIndex
			 * Purpose:  Sorting index which will be used by default
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.iDefaultSortIndex = 0;
			
			/*
			 * Variable: bInitialised
			 * Purpose:  Indicate if all required information has been read in
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.bInitialised = false;
			
			/*
			 * Variable: aoOpenRows
			 * Purpose:  Information about open rows
			 * Scope:    jQuery.dataTable.classSettings
			 * Notes:    Has the parameters 'nTr' and 'nParent'
			 */
			this.aoOpenRows = [];
			
			/*
			 * Variable: sDom
			 * Purpose:  Dictate the positioning that the created elements will take
			 * Scope:    jQuery.dataTable.classSettings
			 * Notes:    
			 *   The following options are allowed:
			 *     'l' - Length changing
			 *     'f' - Filtering input
			 *     't' - The table!
			 *     'i' - Information
			 *     'p' - Pagination
			 *     'r' - pRocessing
			 *   The following constants are allowed:
			 *     'H' - jQueryUI theme "header" classes
			 *     'F' - jQueryUI theme "footer" classes
			 *   The following syntax is expected:
			 *     '<' and '>' - div elements
			 *     '<"class" and '>' - div with a class
			 *   Examples:
			 *     '<"wrapper"flipt>', '<lf<t>ip>'
			 */
			this.sDom = 'lfrtip';
			
			/*
			 * Variable: sPaginationType
			 * Purpose:  Note which type of sorting should be used
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.sPaginationType = "two_button";
			
			/*
			 * Variable: iCookieDuration
			 * Purpose:  The cookie duration (for bStateSave) in seconds - default 2 hours
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.iCookieDuration = 60 * 60 * 2;
			
			/*
			 * Variable: sAjaxSource
			 * Purpose:  Source url for AJAX data for the table
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.sAjaxSource = null;
			
			/*
			 * Variable: bAjaxDataGet
			 * Purpose:  Note if draw should be blocked while getting data
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.bAjaxDataGet = true;
			
			/*
			 * Variable: fnServerData
			 * Purpose:  Function to get the server-side data - can be overruled by the developer
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.fnServerData = $.getJSON;
			
			/*
			 * Variable: iServerDraw
			 * Purpose:  Counter and tracker for server-side processing draws
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.iServerDraw = 0;
			
			/*
			 * Variable: _iDisplayLength, _iDisplayStart, _iDisplayEnd
			 * Purpose:  Display length variables
			 * Scope:    jQuery.dataTable.classSettings
			 * Notes:    These variable must NOT be used externally to get the data length. Rather, use
			 *   the fnRecordsTotal() (etc) functions.
			 */
			this._iDisplayLength = 10;
			this._iDisplayStart = 0;
			this._iDisplayEnd = 10;
			
			/*
			 * Variable: _iRecordsTotal, _iRecordsDisplay
			 * Purpose:  Display length variables used for server side processing
			 * Scope:    jQuery.dataTable.classSettings
			 * Notes:    These variable must NOT be used externally to get the data length. Rather, use
			 *   the fnRecordsTotal() (etc) functions.
			 */
			this._iRecordsTotal = 0;
			this._iRecordsDisplay = 0;
			
			/*
			 * Variable: bJUI
			 * Purpose:  Should we add the markup needed for jQuery UI theming?
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.bJUI = false;
			
			/*
			 * Variable: bJUI
			 * Purpose:  Should we add the markup needed for jQuery UI theming?
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.oClasses = _oExt.oStdClasses;
			
			/*
			 * Variable: bFiltered and bSorted
			 * Purpose:  Flag to allow callback functions to see what action has been performed
			 * Scope:    jQuery.dataTable.classSettings
			 */
			this.bFiltered = false;
			this.bSorted = false;
		}
		
		/*
		 * Variable: oApi
		 * Purpose:  Container for publicly exposed 'private' functions
		 * Scope:    jQuery.dataTable
		 */
		this.oApi = {};
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - API functions
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		/*
		 * Function: fnDraw
		 * Purpose:  Redraw the table
		 * Returns:  -
		 * Inputs:   bool:bComplete - Refilter and resort (if enabled) the table before the draw.
		 *             Optional: default - true
		 */
		this.fnDraw = function( bComplete )
		{
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			if ( typeof bComplete != 'undefined' && bComplete === false )
			{
				_fnCalculateEnd( oSettings );
				_fnDraw( oSettings );
			}
			else
			{
				_fnReDraw( oSettings );
			}
		};
		
		/*
		 * Function: fnFilter
		 * Purpose:  Filter the input based on data
		 * Returns:  -
		 * Inputs:   string:sInput - string to filter the table on
		 *           int:iColumn - optional - column to limit filtering to
		 *           bool:bEscapeRegex - optional - escape regex characters or not - default true
		 */
		this.fnFilter = function( sInput, iColumn, bEscapeRegex )
		{
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			
			if ( typeof bEscapeRegex == 'undefined' )
			{
				bEscapeRegex = true;
			}
			
			if ( typeof iColumn == "undefined" || iColumn === null )
			{
				/* Global filter */
				_fnFilterComplete( oSettings, {"sSearch":sInput, "bEscapeRegex": bEscapeRegex}, 1 );
			}
			else
			{
				/* Single column filter */
				oSettings.aoPreSearchCols[ iColumn ].sSearch = sInput;
				oSettings.aoPreSearchCols[ iColumn ].bEscapeRegex = bEscapeRegex;
				_fnFilterComplete( oSettings, oSettings.oPreviousSearch, 1 );
			}
		};
		
		/*
		 * Function: fnSettings
		 * Purpose:  Get the settings for a particular table for extern. manipulation
		 * Returns:  -
		 * Inputs:   -
		 */
		this.fnSettings = function( nNode  )
		{
			return _fnSettingsFromNode( this[_oExt.iApiIndex] );
		};
		
		/*
		 * Function: fnVersionCheck
		 * Purpose:  Check a version string against this version of DataTables. Useful for plug-ins
		 * Returns:  bool:true -this version of DataTables is greater or equal to the required version
		 *                false -this version of DataTales is not suitable
		 * Inputs:   string:sVersion - the version to check against. May be in the following formats:
		 *             "a", "a.b" or "a.b.c"
		 * Notes:    This function will only check the first three parts of a version string. It is
		 *   assumed that beta and dev versions will meet the requirements. This might change in future
		 */
		this.fnVersionCheck = function( sVersion )
		{
			/* This is cheap, but very effective */
			var fnZPad = function (Zpad, count)
			{
				while(Zpad.length < count) {
					Zpad += '0';
				}
				return Zpad;
			};
			var aThis = _oExt.sVersion.split('.');
			var aThat = sVersion.split('.');
			var sThis = '', sThat = '';
			
			for ( var i=0, iLen=aThat.length ; i<iLen ; i++ )
			{
				sThis += fnZPad( aThis[i], 3 );
				sThat += fnZPad( aThat[i], 3 );
			}
			
			return parseInt(sThis, 10) >= parseInt(sThat, 10);
		};
		
		/*
		 * Function: fnSort
		 * Purpose:  Sort the table by a particular row
		 * Returns:  -
		 * Inputs:   int:iCol - the data index to sort on. Note that this will
		 *   not match the 'display index' if you have hidden data entries
		 */
		this.fnSort = function( aaSort )
		{
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			oSettings.aaSorting = aaSort;
			_fnSort( oSettings );
		};
		
		/*
		 * Function: fnSortListener
		 * Purpose:  Attach a sort listener to an element for a given column
		 * Returns:  -
		 * Inputs:   node:nNode - the element to attach the sort listener to
		 *           int:iColumn - the column that a click on this node will sort on
		 *           function:fnCallback - callback function when sort is run - optional
		 */
		this.fnSortListener = function( nNode, iColumn, fnCallback )
		{
			_fnSortAttachListener( _fnSettingsFromNode( this[_oExt.iApiIndex] ), nNode, iColumn,
			 	fnCallback );
		};
		
		/*
		 * Function: fnAddData
		 * Purpose:  Add new row(s) into the table
		 * Returns:  array int: array of indexes (aoData) which have been added (zero length on error)
		 * Inputs:   array:mData - the data to be added. The length must match
		 *               the original data from the DOM
		 *             or
		 *             array array:mData - 2D array of data to be added
		 *           bool:bRedraw - redraw the table or not - default true
		 * Notes:    Warning - the refilter here will cause the table to redraw
		 *             starting at zero
		 * Notes:    Thanks to Yekimov Denis for contributing the basis for this function!
		 */
		this.fnAddData = function( mData, bRedraw )
		{
			if ( mData.length === 0 )
			{
				return [];
			}
			
			var aiReturn = [];
			var iTest;
			
			/* Find settings from table node */
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			
			/* Check if we want to add multiple rows or not */
			if ( typeof mData[0] == "object" )
			{
				for ( var i=0 ; i<mData.length ; i++ )
				{
					iTest = _fnAddData( oSettings, mData[i] );
					if ( iTest == -1 )
					{
						return aiReturn;
					}
					aiReturn.push( iTest );
				}
			}
			else
			{
				iTest = _fnAddData( oSettings, mData );
				if ( iTest == -1 )
				{
					return aiReturn;
				}
				aiReturn.push( iTest );
			}
			
			oSettings.aiDisplay = oSettings.aiDisplayMaster.slice();
			
			/* Rebuild the search */
			_fnBuildSearchArray( oSettings, 1 );
			
			if ( typeof bRedraw == 'undefined' || bRedraw )
			{
				_fnReDraw( oSettings );
			}
			return aiReturn;
		};
		
		/*
		 * Function: fnDeleteRow
		 * Purpose:  Remove a row for the table
		 * Returns:  array:aReturn - the row that was deleted
		 * Inputs:   mixed:mTarget - 
		 *             int: - index of aoData to be deleted, or
		 *             node(TR): - TR element you want to delete
		 *           function:fnCallBack - callback function - default null
		 *           bool:bNullRow - remove the row information from aoData by setting the value to
		 *             null - default false
		 * Notes:    This function requires a little explanation - we don't actually delete the data
		 *   from aoData - rather we remove it's references from aiDisplayMastr and aiDisplay. This
		 *   in effect prevnts DataTables from drawing it (hence deleting it) - it could be restored
		 *   if you really wanted. The reason for this is that actually removing the aoData object
		 *   would mess up all the subsequent indexes in the display arrays (they could be ajusted - 
		 *   but this appears to do what is required).
		 */
		this.fnDeleteRow = function( mTarget, fnCallBack, bNullRow )
		{
			/* Find settings from table node */
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			var i, iAODataIndex;
			
			iAODataIndex = (typeof mTarget == 'object') ? 
				_fnNodeToDataIndex(oSettings, mTarget) : mTarget;
			
			/* Delete from the display master */
			for ( i=0 ; i<oSettings.aiDisplayMaster.length ; i++ )
			{
				if ( oSettings.aiDisplayMaster[i] == iAODataIndex )
				{
					oSettings.aiDisplayMaster.splice( i, 1 );
					break;
				}
			}
			
			/* Delete from the current display index */
			for ( i=0 ; i<oSettings.aiDisplay.length ; i++ )
			{
				if ( oSettings.aiDisplay[i] == iAODataIndex )
				{
					oSettings.aiDisplay.splice( i, 1 );
					break;
				}
			}
			
			/* Rebuild the search */
			_fnBuildSearchArray( oSettings, 1 );
			
			/* If there is a user callback function - call it */
			if ( typeof fnCallBack == "function" )
			{
				fnCallBack.call( this );
			}
			
			/* Check for an 'overflow' they case for dislaying the table */
			if ( oSettings._iDisplayStart >= oSettings.aiDisplay.length )
			{
				oSettings._iDisplayStart -= oSettings._iDisplayLength;
				if ( oSettings._iDisplayStart < 0 )
				{
					oSettings._iDisplayStart = 0;
				}
			}
			
			_fnCalculateEnd( oSettings );
			_fnDraw( oSettings );
			
			/* Return the data array from this row */
			var aData = oSettings.aoData[iAODataIndex]._aData.slice();
			
			if ( typeof bNullRow != "undefined" && bNullRow === true )
			{
				oSettings.aoData[iAODataIndex] = null;
			}
			
			return aData;
		};
		
		/*
		 * Function: fnClearTable
		 * Purpose:  Quickly and simply clear a table
		 * Returns:  -
		 * Inputs:   bool:bRedraw - redraw the table or not - default true
		 * Notes:    Thanks to Yekimov Denis for contributing the basis for this function!
		 */
		this.fnClearTable = function( bRedraw )
		{
			/* Find settings from table node */
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			_fnClearTable( oSettings );
			
			if ( typeof bRedraw == 'undefined' || bRedraw )
			{
				_fnDraw( oSettings );
			}
		};
		
		/*
		 * Function: fnOpen
		 * Purpose:  Open a display row (append a row after the row in question)
		 * Returns:  node:nNewRow - the row opened
		 * Inputs:   node:nTr - the table row to 'open'
		 *           string:sHtml - the HTML to put into the row
		 *           string:sClass - class to give the new cell
		 */
		this.fnOpen = function( nTr, sHtml, sClass )
		{
			/* Find settings from table node */
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			
			/* the old open one if there is one */
			this.fnClose( nTr );
			
			var nNewRow = document.createElement("tr");
			var nNewCell = document.createElement("td");
			nNewRow.appendChild( nNewCell );
			nNewCell.className = sClass;
			nNewCell.colSpan = _fnVisbleColumns( oSettings );
			nNewCell.innerHTML = sHtml;
			
			/* If the nTr isn't on the page at the moment - then we don't insert at the moment */
			var nTrs = $('tbody tr', oSettings.nTable);
			if ( $.inArray(nTr, nTrs) != -1 )
			{
				$(nNewRow).insertAfter(nTr);
			}
			
			/* No point in storing the row if using server-side processing since the nParent will be
			 * nuked on a re-draw anyway
			 */
			if ( !oSettings.oFeatures.bServerSide )
			{
				oSettings.aoOpenRows.push( {
					"nTr": nNewRow,
					"nParent": nTr
				} );
			}
			
			return nNewRow;
		};
		
		/*
		 * Function: fnClose
		 * Purpose:  Close a display row
		 * Returns:  int: 0 (success) or 1 (failed)
		 * Inputs:   node:nTr - the table row to 'close'
		 */
		this.fnClose = function( nTr )
		{
			/* Find settings from table node */
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			
			for ( var i=0 ; i<oSettings.aoOpenRows.length ; i++ )
			{
				if ( oSettings.aoOpenRows[i].nParent == nTr )
				{
					var nTrParent = oSettings.aoOpenRows[i].nTr.parentNode;
					if ( nTrParent )
					{
						/* Remove it if it is currently on display */
						nTrParent.removeChild( oSettings.aoOpenRows[i].nTr );
					}
					oSettings.aoOpenRows.splice( i, 1 );
					return 0;
				}
			}
			return 1;
		};
		
		/*
		 * Function: fnGetData
		 * Purpose:  Return an array with the data which is used to make up the table
		 * Returns:  array array string: 2d data array ([row][column]) or array string: 1d data array
		 *           or
		 *           array string (if iRow specified)
		 * Inputs:   mixed:mRow - optional - if not present, then the full 2D array for the table 
		 *             if given then:
		 *               int: - return 1D array for aoData entry of this index
		 *               node(TR): - return 1D array for this TR element
		 * Inputs:   int:iRow - optional - if present then the array returned will be the data for
		 *             the row with the index 'iRow'
		 */
		this.fnGetData = function( mRow )
		{
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			
			if ( typeof mRow != 'undefined' )
			{
				var iRow = (typeof mRow == 'object') ? 
					_fnNodeToDataIndex(oSettings, mRow) : mRow;
				return oSettings.aoData[iRow]._aData;
			}
			return _fnGetDataMaster( oSettings );
		};
		
		/*
		 * Function: fnGetNodes
		 * Purpose:  Return an array with the TR nodes used for drawing the table
		 * Returns:  array node: TR elements
		 *           or
		 *           node (if iRow specified)
		 * Inputs:   int:iRow - optional - if present then the array returned will be the node for
		 *             the row with the index 'iRow'
		 */
		this.fnGetNodes = function( iRow )
		{
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			
			if ( typeof iRow != 'undefined' )
			{
				return oSettings.aoData[iRow].nTr;
			}
			return _fnGetTrNodes( oSettings );
		};
		
		/*
		 * Function: fnGetPosition
		 * Purpose:  Get the array indexes of a particular cell from it's DOM element
		 * Returns:  int: - row index, or array[ int, int, int ]: - row index, column index (visible)
		 *             and column index including hidden columns
		 * Inputs:   node:nNode - this can either be a TR or a TD in the table, the return is
		 *             dependent on this input
		 */
		this.fnGetPosition = function( nNode )
		{
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			var i;
			
			if ( nNode.nodeName == "TR" )
			{
				return _fnNodeToDataIndex(oSettings, nNode);
			}
			else if ( nNode.nodeName == "TD" )
			{
				var iDataIndex = _fnNodeToDataIndex(oSettings, nNode.parentNode);
				var iCorrector = 0;
				for ( var j=0 ; j<oSettings.aoColumns.length ; j++ )
				{
					if ( oSettings.aoColumns[j].bVisible )
					{
						if ( oSettings.aoData[iDataIndex].nTr.getElementsByTagName('td')[j-iCorrector] == nNode )
						{
							return [ iDataIndex, j-iCorrector, j ];
						}
					}
					else
					{
						iCorrector++;
					}
				}
			}
			return null;
		};
		
		/*
		 * Function: fnUpdate
		 * Purpose:  Update a table cell or row
		 * Returns:  int: 0 okay, 1 error
		 * Inputs:   array string 'or' string:mData - data to update the cell/row with
		 *           mixed:mRow - 
		 *             int: - index of aoData to be updated, or
		 *             node(TR): - TR element you want to update
		 *           int:iColumn - the column to update - optional (not used of mData is 2D)
		 *           bool:bRedraw - redraw the table or not - default true
		 */
		this.fnUpdate = function( mData, mRow, iColumn, bRedraw )
		{
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			var iVisibleColumn;
			var sDisplay;
			var iRow = (typeof mRow == 'object') ? 
				_fnNodeToDataIndex(oSettings, mRow) : mRow;
			
			if ( typeof mData != 'object' )
			{
				sDisplay = mData;
				oSettings.aoData[iRow]._aData[iColumn] = sDisplay;
				
				if ( oSettings.aoColumns[iColumn].fnRender !== null )
				{
					sDisplay = oSettings.aoColumns[iColumn].fnRender( {
						"iDataRow": iRow,
						"iDataColumn": iColumn,
						"aData": oSettings.aoData[iRow]._aData,
						"oSettings": oSettings
					} );
					
					if ( oSettings.aoColumns[iColumn].bUseRendered )
					{
						oSettings.aoData[iRow]._aData[iColumn] = sDisplay;
					}
				}
				
				iVisibleColumn = _fnColumnIndexToVisible( oSettings, iColumn );
				if ( iVisibleColumn !== null )
				{
					oSettings.aoData[iRow].nTr.getElementsByTagName('td')[iVisibleColumn].innerHTML = 
						sDisplay;
				}
			}
			else
			{
				if ( mData.length != oSettings.aoColumns.length )
				{
					alert( 'DataTables warning: An array passed to fnUpdate must have the same number of '+
						'columns as the table in question - in this case '+oSettings.aoColumns.length );
					return 1;
				}
				
				for ( var i=0 ; i<mData.length ; i++ )
				{
					sDisplay = mData[i];
					oSettings.aoData[iRow]._aData[i] = sDisplay;
					
					if ( oSettings.aoColumns[i].fnRender !== null )
					{
						sDisplay = oSettings.aoColumns[i].fnRender( {
							"iDataRow": iRow,
							"iDataColumn": i,
							"aData": oSettings.aoData[iRow]._aData,
							"oSettings": oSettings
						} );
						
						if ( oSettings.aoColumns[i].bUseRendered )
						{
							oSettings.aoData[iRow]._aData[i] = sDisplay;
						}
					}
					
					iVisibleColumn = _fnColumnIndexToVisible( oSettings, i );
					if ( iVisibleColumn !== null )
					{
						oSettings.aoData[iRow].nTr.getElementsByTagName('td')[iVisibleColumn].innerHTML = 
							sDisplay;
					}
				}
			}
			
			/* Update the search array */
			_fnBuildSearchArray( oSettings, 1 );
			
			/* Redraw the table */
			if ( typeof bRedraw != 'undefined' && bRedraw )
			{
				_fnReDraw( oSettings );
			}
			return 0;
		};
		
		
		/*
		 * Function: fnShowColoumn
		 * Purpose:  Show a particular column
		 * Returns:  -
		 * Inputs:   int:iCol - the column whose display should be changed
		 *           bool:bShow - show (true) or hide (false) the column
		 */
		this.fnSetColumnVis = function ( iCol, bShow )
		{
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			var i, iLen;
			var iColumns = oSettings.aoColumns.length;
			var nTd, anTds;
			
			/* No point in doing anything if we are requesting what is already true */
			if ( oSettings.aoColumns[iCol].bVisible == bShow )
			{
				return;
			}
			
			var nTrHead = $('thead:eq(0)>tr', oSettings.nTable)[0];
			var nTrFoot = $('tfoot:eq(0)>tr', oSettings.nTable)[0];
			var anTheadTh = [];
			var anTfootTh = [];
			for ( i=0 ; i<iColumns ; i++ )
			{
				anTheadTh.push( oSettings.aoColumns[i].nTh );
				anTfootTh.push( oSettings.aoColumns[i].nTf );
			}
			
			/* Show the column */
			if ( bShow )
			{
				var iInsert = 0;
				for ( i=0 ; i<iCol ; i++ )
				{
					if ( oSettings.aoColumns[i].bVisible )
					{
						iInsert++;
					}
				}
				
				/* Need to decide if we should use appendChild or insertBefore */
				if ( iInsert >= _fnVisbleColumns( oSettings ) )
				{
					nTrHead.appendChild( anTheadTh[iCol] );
					if ( nTrFoot )
					{
						nTrFoot.appendChild( anTfootTh[iCol] );
					}
					
					for ( i=0, iLen=oSettings.aoData.length ; i<iLen ; i++ )
					{
						nTd = oSettings.aoData[i]._anHidden[iCol];
						oSettings.aoData[i].nTr.appendChild( nTd );
					}
				}
				else
				{
					/* Which coloumn should we be inserting before? */
					var iBefore;
					for ( i=iCol ; i<iColumns ; i++ )
					{
						iBefore = _fnColumnIndexToVisible( oSettings, i );
						if ( iBefore !== null )
						{
							break;
						}
					}
					
					nTrHead.insertBefore( anTheadTh[iCol], nTrHead.getElementsByTagName('th')[iBefore] );
					if ( nTrFoot )
					{
						nTrFoot.insertBefore( anTfootTh[iCol], nTrFoot.getElementsByTagName('th')[iBefore] );
					}
					
					anTds = _fnGetTdNodes( oSettings );
					for ( i=0, iLen=oSettings.aoData.length ; i<iLen ; i++ )
					{
						nTd = oSettings.aoData[i]._anHidden[iCol];
						oSettings.aoData[i].nTr.insertBefore( nTd, $('>td:eq('+iBefore+')', 
							oSettings.aoData[i].nTr)[0] );
					}
				}
				
				oSettings.aoColumns[iCol].bVisible = true;
			}
			else
			{
				/* Remove a column from display */
				nTrHead.removeChild( anTheadTh[iCol] );
				if ( nTrFoot )
				{
					nTrFoot.removeChild( anTfootTh[iCol] );
				}
				
				anTds = _fnGetTdNodes( oSettings );
				for ( i=0, iLen=oSettings.aoData.length ; i<iLen ; i++ )
				{
					nTd = anTds[ ( i*oSettings.aoColumns.length) + iCol ];
					oSettings.aoData[i]._anHidden[iCol] = nTd;
					nTd.parentNode.removeChild( nTd );
				}
				
				oSettings.aoColumns[iCol].bVisible = false;
			}
			
			/* If there are any 'open' rows, then we need to alter the colspan for this col change */
			for ( i=0, iLen=oSettings.aoOpenRows.length ; i<iLen ; i++ )
			{
				oSettings.aoOpenRows[i].nTr.colSpan = _fnVisbleColumns( oSettings );
			}
			
			/* Since there is no redraw done here, we need to save the state manually */
			_fnSaveState( oSettings );
		};
		
		/*
		 * Function: fnPageChange
		 * Purpose:  Change the pagination
		 * Returns:  -
		 * Inputs:   string:sAction - paging action to take: "first", "previous", "next" or "last"
		 *           bool:bRedraw - redraw the table or not - optional - default true
		 */
		this.fnPageChange = function ( sAction, bRedraw )
		{
			var oSettings = _fnSettingsFromNode( this[_oExt.iApiIndex] );
			_fnPageChange( oSettings, sAction );
			_fnCalculateEnd( oSettings );
			
			if ( typeof bRedraw == 'undefined' || bRedraw )
			{
				_fnDraw( oSettings );
			}
		};
		
		
		/*
		 * Plugin API functions
		 * 
		 * This call will add the functions which are defined in _oExt.oApi to the
		 * DataTables object, providing a rather nice way to allow plug-in API functions. Note that
		 * this is done here, so that API function can actually override the built in API functions if
		 * required for a particular purpose.
		 */
		
		/*
		 * Function: _fnExternApiFunc
		 * Purpose:  Create a wrapper function for exporting an internal func to an external API func
		 * Returns:  function: - wrapped function
		 * Inputs:   string:sFunc - API function name
		 */
		function _fnExternApiFunc (sFunc)
		{
			return function() {
					var aArgs = [_fnSettingsFromNode(this[_oExt.iApiIndex])].concat( 
						Array.prototype.slice.call(arguments) );
					return _oExt.oApi[sFunc].apply( this, aArgs );
				};
		}
		
		for ( var sFunc in _oExt.oApi )
		{
			if ( sFunc )
			{
				/*
				 * Function: anon
				 * Purpose:  Wrap the plug-in API functions in order to provide the settings as 1st arg 
				 *   and execute in this scope
				 * Returns:  -
				 * Inputs:   -
				 */
				this[sFunc] = _fnExternApiFunc(sFunc);
			}
		}
		
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Local functions
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Initalisation
		 */
		
		/*
		 * Function: _fnInitalise
		 * Purpose:  Draw the table for the first time, adding all required features
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnInitalise ( oSettings )
		{
			/* Ensure that the table data is fully initialised */
			if ( oSettings.bInitialised === false )
			{
				setTimeout( function(){ _fnInitalise( oSettings ); }, 200 );
				return;
			}
			
			/* Show the display HTML options */
			_fnAddOptionsHtml( oSettings );
			
			/* Draw the headers for the table */
			_fnDrawHead( oSettings );
			
			/* If there is default sorting required - let's do it. The sort function
			 * will do the drawing for us. Otherwise we draw the table
			 */
			if ( oSettings.oFeatures.bSort )
			{
				_fnSort( oSettings, false );
				/*
				 * Add the sorting classes to the header and the body (if needed).
				 * Reason for doing it here after the first draw is to stop classes being applied to the
				 * 'static' table.
				 */
				_fnSortingClasses( oSettings );
			}
			else
			{
				oSettings.aiDisplay = oSettings.aiDisplayMaster.slice();
				_fnCalculateEnd( oSettings );
				_fnDraw( oSettings );
			}
			
			/* if there is an ajax source */
			if ( oSettings.sAjaxSource !== null && !oSettings.oFeatures.bServerSide )
			{
				_fnProcessingDisplay( oSettings, true );
				
				oSettings.fnServerData( oSettings.sAjaxSource, null, function(json) {
					
					/* Got the data - add it to the table */
					for ( var i=0 ; i<json.aaData.length ; i++ )
					{
						_fnAddData( oSettings, json.aaData[i] );
					}
					
					/* Reset the init display for cookie saving. We've already done a filter, and
					 * therefore cleared it before. So we need to make it appear 'fresh'
					 */
					oSettings.iInitDisplayStart = oSettings._iDisplayStart;
					
					if ( oSettings.oFeatures.bSort )
					{
						_fnSort( oSettings );
					}
					else
					{
						oSettings.aiDisplay = oSettings.aiDisplayMaster.slice();
						_fnCalculateEnd( oSettings );
						_fnDraw( oSettings );
					}
					
					_fnProcessingDisplay( oSettings, false );
					
					/* Run the init callback if there is one */
					if ( typeof oSettings.fnInitComplete == 'function' )
					{
						oSettings.fnInitComplete( oSettings, json );
					}
				} );
				return;
			}
			
			/* Run the init callback if there is one */
			if ( typeof oSettings.fnInitComplete == 'function' )
			{
				oSettings.fnInitComplete( oSettings );
			}
			
			if ( !oSettings.oFeatures.bServerSide )
			{
				_fnProcessingDisplay( oSettings, false );
			}
		}
		
		/*
		 * Function: _fnLanguageProcess
		 * Purpose:  Copy language variables from remote object to a local one
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 *           object:oLanguage - Language information
		 *           bool:bInit - init once complete
		 */
		function _fnLanguageProcess( oSettings, oLanguage, bInit )
		{
			_fnMap( oSettings.oLanguage, oLanguage, 'sProcessing' );
			_fnMap( oSettings.oLanguage, oLanguage, 'sLengthMenu' );
			_fnMap( oSettings.oLanguage, oLanguage, 'sZeroRecords' );
			_fnMap( oSettings.oLanguage, oLanguage, 'sInfo' );
			_fnMap( oSettings.oLanguage, oLanguage, 'sInfoEmpty' );
			_fnMap( oSettings.oLanguage, oLanguage, 'sInfoFiltered' );
			_fnMap( oSettings.oLanguage, oLanguage, 'sInfoPostFix' );
			_fnMap( oSettings.oLanguage, oLanguage, 'sSearch' );
			
			if ( typeof oLanguage.oPaginate != 'undefined' )
			{
				_fnMap( oSettings.oLanguage.oPaginate, oLanguage.oPaginate, 'sFirst' );
				_fnMap( oSettings.oLanguage.oPaginate, oLanguage.oPaginate, 'sPrevious' );
				_fnMap( oSettings.oLanguage.oPaginate, oLanguage.oPaginate, 'sNext' );
				_fnMap( oSettings.oLanguage.oPaginate, oLanguage.oPaginate, 'sLast' );
			}
			
			if ( bInit )
			{
				_fnInitalise( oSettings );
			}
		}
		
		/*
		 * Function: _fnAddColumn
		 * Purpose:  Add a column to the list used for the table
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 *           object:oOptions - object with sType, bVisible and bSearchable
		 *           node:nTh - the th element for this column
		 * Notes:    All options in enter column can be over-ridden by the user
		 *   initialisation of dataTables
		 */
		function _fnAddColumn( oSettings, oOptions, nTh )
		{
			oSettings.aoColumns[ oSettings.aoColumns.length++ ] = {
				"sType": null,
				"_bAutoType": true,
				"bVisible": true,
				"bSearchable": true,
				"bSortable": true,
				"asSorting": [ 'asc', 'desc' ],
				"sSortingClass": oSettings.oClasses.sSortable,
				"sSortingClassJUI": oSettings.oClasses.sSortJUI,
				"sTitle": nTh ? nTh.innerHTML : '',
				"sName": '',
				"sWidth": null,
				"sClass": null,
				"fnRender": null,
				"bUseRendered": true,
				"iDataSort": oSettings.aoColumns.length-1,
				"sSortDataType": 'std',
				"nTh": nTh ? nTh : document.createElement('th'),
				"nTf": null
			};
			
			var iLength = oSettings.aoColumns.length-1;
			var oCol = oSettings.aoColumns[ iLength ];
			
			/* User specified column options */
			if ( typeof oOptions != 'undefined' && oOptions !== null )
			{
				if ( typeof oOptions.sType != 'undefined' )
				{
					oCol.sType = oOptions.sType;
					oCol._bAutoType = false;
				}
				
				_fnMap( oCol, oOptions, "bVisible" );
				_fnMap( oCol, oOptions, "bSearchable" );
				_fnMap( oCol, oOptions, "bSortable" );
				_fnMap( oCol, oOptions, "sTitle" );
				_fnMap( oCol, oOptions, "sName" );
				_fnMap( oCol, oOptions, "sWidth" );
				_fnMap( oCol, oOptions, "sClass" );
				_fnMap( oCol, oOptions, "fnRender" );
				_fnMap( oCol, oOptions, "bUseRendered" );
				_fnMap( oCol, oOptions, "iDataSort" );
				_fnMap( oCol, oOptions, "asSorting" );
				_fnMap( oCol, oOptions, "sSortDataType" );
			}
			
			/* Feature sorting overrides column specific when off */
			if ( !oSettings.oFeatures.bSort )
			{
				oCol.bSortable = false;
			}
			
			/* Check that the class assignment is correct for sorting */
			if ( !oCol.bSortable ||
					 ($.inArray('asc', oCol.asSorting) == -1 && $.inArray('desc', oCol.asSorting) == -1) )
			{
				oCol.sSortingClass = oSettings.oClasses.sSortableNone;
				oCol.sSortingClassJUI = "";
			}
			else if ( $.inArray('asc', oCol.asSorting) != -1 && $.inArray('desc', oCol.asSorting) == -1 )
			{
				oCol.sSortingClass = oSettings.oClasses.sSortableAsc;
				oCol.sSortingClassJUI = oSettings.oClasses.sSortJUIAscAllowed;
			}
			else if ( $.inArray('asc', oCol.asSorting) == -1 && $.inArray('desc', oCol.asSorting) != -1 )
			{
				oCol.sSortingClass = oSettings.oClasses.sSortableDesc;
				oCol.sSortingClassJUI = oSettings.oClasses.sSortJUIDescAllowed;
			}
			
			/* Add a column specific filter */
			if ( typeof oSettings.aoPreSearchCols[ iLength ] == 'undefined' ||
			     oSettings.aoPreSearchCols[ iLength ] === null )
			{
				oSettings.aoPreSearchCols[ iLength ] = {
					"sSearch": "",
					"bEscapeRegex": true
				};
			}
			else if ( typeof oSettings.aoPreSearchCols[ iLength ].bEscapeRegex == 'undefined' )
			{
				/* Don't require that the user must specify bEscapeRegex */
				oSettings.aoPreSearchCols[ iLength ].bEscapeRegex = true;
			}
		}
		
		/*
		 * Function: _fnAddData
		 * Purpose:  Add a data array to the table, creating DOM node etc
		 * Returns:  int: - >=0 if successful (index of new aoData entry), -1 if failed
		 * Inputs:   object:oSettings - dataTables settings object
		 *           array:aData - data array to be added
		 */
		function _fnAddData ( oSettings, aData )
		{
			/* Sanity check the length of the new array */
			if ( aData.length != oSettings.aoColumns.length )
			{
				alert( "DataTables warning: Added data does not match known number of columns" );
				return -1;
			}
			
			/* Create the object for storing information about this new row */
			var iThisIndex = oSettings.aoData.length;
			oSettings.aoData.push( {
				"nTr": document.createElement('tr'),
				"_iId": oSettings.iNextId++,
				"_aData": aData.slice(),
				"_anHidden": [],
				"_sRowStripe": ''
			} );
			
			/* Create the cells */
			var nTd, sThisType;
			for ( var i=0 ; i<aData.length ; i++ )
			{
				nTd = document.createElement('td');
				
				if ( typeof oSettings.aoColumns[i].fnRender == 'function' )
				{
					var sRendered = oSettings.aoColumns[i].fnRender( {
							"iDataRow": iThisIndex,
							"iDataColumn": i,
							"aData": aData,
							"oSettings": oSettings
						} );
					nTd.innerHTML = sRendered;
					if ( oSettings.aoColumns[i].bUseRendered )
					{
						/* Use the rendered data for filtering/sorting */
						oSettings.aoData[iThisIndex]._aData[i] = sRendered;
					}
				}
				else
				{
					nTd.innerHTML = aData[i];
				}
				
				if ( oSettings.aoColumns[i].sClass !== null )
				{
					nTd.className = oSettings.aoColumns[i].sClass;
				}
				
				/* See if we should auto-detect the column type */
				if ( oSettings.aoColumns[i]._bAutoType && oSettings.aoColumns[i].sType != 'string' )
				{
					/* Attempt to auto detect the type - same as _fnGatherData() */
					sThisType = _fnDetectType( oSettings.aoData[iThisIndex]._aData[i] );
					if ( oSettings.aoColumns[i].sType === null )
					{
						oSettings.aoColumns[i].sType = sThisType;
					}
					else if ( oSettings.aoColumns[i].sType != sThisType )
					{
						/* String is always the 'fallback' option */
						oSettings.aoColumns[i].sType = 'string';
					}
				}
					
				if ( oSettings.aoColumns[i].bVisible )
				{
					oSettings.aoData[iThisIndex].nTr.appendChild( nTd );
				}
				else
				{
					oSettings.aoData[iThisIndex]._anHidden[i] = nTd;
				}
			}
			
			/* Add to the display array */
			oSettings.aiDisplayMaster.push( iThisIndex );
			return iThisIndex;
		}
		
		/*
		 * Function: _fnGatherData
		 * Purpose:  Read in the data from the target table
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnGatherData( oSettings )
		{
			var iLoop, i, iLen, j, jLen, jInner,
			 	nTds, nTrs, nTd, aLocalData, iThisIndex,
				iRow, iRows, iColumn, iColumns;
			
			/*
			 * Process by row first
			 * Add the data object for the whole table - storing the tr node. Note - no point in getting
			 * DOM based data if we are going to go and replace it with Ajax source data.
			 */
			if ( oSettings.sAjaxSource === null )
			{
				nTrs = oSettings.nTable.getElementsByTagName('tbody')[0].childNodes;
				for ( i=0, iLen=nTrs.length ; i<iLen ; i++ )
				{
					if ( nTrs[i].nodeName == "TR" )
					{
						iThisIndex = oSettings.aoData.length;
						oSettings.aoData.push( {
							"nTr": nTrs[i],
							"_iId": oSettings.iNextId++,
							"_aData": [],
							"_anHidden": [],
							"_sRowStripe": ''
						} );
						
						oSettings.aiDisplayMaster.push( iThisIndex );
						
						aLocalData = oSettings.aoData[iThisIndex]._aData;
						nTds = nTrs[i].childNodes;
						jInner = 0;
						
						for ( j=0, jLen=nTds.length ; j<jLen ; j++ )
						{
							if ( nTds[j].nodeName == "TD" )
							{
								aLocalData[jInner] = nTds[j].innerHTML;
								jInner++;
							}
						}
					}
				}
			}
			
			/* Gather in the TD elements of the Table - note that this is basically the same as
			 * fnGetTdNodes, but that function takes account of hidden columns, which we haven't yet
			 * setup!
			 */
			nTrs = _fnGetTrNodes( oSettings );
			nTds = [];
			for ( i=0, iLen=nTrs.length ; i<iLen ; i++ )
			{
				for ( j=0, jLen=nTrs[i].childNodes.length ; j<jLen ; j++ )
				{
					nTd = nTrs[i].childNodes[j];
					if ( nTd.nodeName == "TD" )
					{
						nTds.push( nTd );
					}
				}
			}
			
			/* Sanity check */
			if ( nTds.length != nTrs.length * oSettings.aoColumns.length )
			{
				alert( "DataTables warning: Unexpected number of TD elements. Expected "+
					(nTrs.length * oSettings.aoColumns.length)+" and got "+nTds.length+". DataTables does "+
					"not support rowspan / colspan in the table body, and there must be one cell for each "+
					"row/column combination." );
			}
			
			/* Now process by column */
			for ( iColumn=0, iColumns=oSettings.aoColumns.length ; iColumn<iColumns ; iColumn++ )
			{
				/* Get the title of the column - unless there is a user set one */
				if ( oSettings.aoColumns[iColumn].sTitle === null )
				{
					oSettings.aoColumns[iColumn].sTitle = oSettings.aoColumns[iColumn].nTh.innerHTML;
				}
				
				var
					bAutoType = oSettings.aoColumns[iColumn]._bAutoType,
					bRender = typeof oSettings.aoColumns[iColumn].fnRender == 'function',
					bClass = oSettings.aoColumns[iColumn].sClass !== null,
					bVisible = oSettings.aoColumns[iColumn].bVisible,
					nCell, sThisType, sRendered;
				
				/* A single loop to rule them all (and be more efficient) */
				if ( bAutoType || bRender || bClass || !bVisible )
				{
					for ( iRow=0, iRows=oSettings.aoData.length ; iRow<iRows ; iRow++ )
					{
						nCell = nTds[ (iRow*iColumns) + iColumn ];
						
						/* Type detection */
						if ( bAutoType )
						{
							if ( oSettings.aoColumns[iColumn].sType != 'string' )
							{
								sThisType = _fnDetectType( oSettings.aoData[iRow]._aData[iColumn] );
								if ( oSettings.aoColumns[iColumn].sType === null )
								{
									oSettings.aoColumns[iColumn].sType = sThisType;
								}
								else if ( oSettings.aoColumns[iColumn].sType != sThisType )
								{
									/* String is always the 'fallback' option */
									oSettings.aoColumns[iColumn].sType = 'string';
								}
							}
						}
						
						/* Rendering */
						if ( bRender )
						{
							sRendered = oSettings.aoColumns[iColumn].fnRender( {
									"iDataRow": iRow,
									"iDataColumn": iColumn,
									"aData": oSettings.aoData[iRow]._aData,
									"oSettings": oSettings
								} );
							nCell.innerHTML = sRendered;
							if ( oSettings.aoColumns[iColumn].bUseRendered )
							{
								/* Use the rendered data for filtering/sorting */
								oSettings.aoData[iRow]._aData[iColumn] = sRendered;
							}
						}
						
						/* Classes */
						if ( bClass )
						{
							nCell.className += ' '+oSettings.aoColumns[iColumn].sClass;
						}
						
						/* Column visability */
						if ( !bVisible )
						{
							oSettings.aoData[iRow]._anHidden[iColumn] = nCell;
							nCell.parentNode.removeChild( nCell );
						}
					}
				}
			}
		}
		
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Drawing functions
		 */
		
		/*
		 * Function: _fnDrawHead
		 * Purpose:  Create the HTML header for the table
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnDrawHead( oSettings )
		{
			var i, nTh, iLen;
			var iThs = oSettings.nTable.getElementsByTagName('thead')[0].getElementsByTagName('th').length;
			var iCorrector = 0;
			
			/* If there is a header in place - then use it - otherwise it's going to get nuked... */
			if ( iThs !== 0 )
			{
				/* We've got a thead from the DOM, so remove hidden columns and apply width to vis cols */
				for ( i=0, iLen=oSettings.aoColumns.length ; i<iLen ; i++ )
				{
					//oSettings.aoColumns[i].nTh = nThs[i];
					nTh = oSettings.aoColumns[i].nTh;
					
					if ( oSettings.aoColumns[i].bVisible )
					{
						/* Set width */
						if ( oSettings.aoColumns[i].sWidth !== null )
						{
							nTh.style.width = oSettings.aoColumns[i].sWidth;
						}
						
						/* Set the title of the column if it is user defined (not what was auto detected) */
						if ( oSettings.aoColumns[i].sTitle != nTh.innerHTML )
						{
							nTh.innerHTML = oSettings.aoColumns[i].sTitle;
						}
					}
					else
					{
						nTh.parentNode.removeChild( nTh );
						iCorrector++;
					}
				}
			}
			else
			{
				/* We don't have a header in the DOM - so we are going to have to create one */
				var nTr = document.createElement( "tr" );
				
				for ( i=0, iLen=oSettings.aoColumns.length ; i<iLen ; i++ )
				{
					nTh = oSettings.aoColumns[i].nTh;
					nTh.innerHTML = oSettings.aoColumns[i].sTitle;
					
					if ( oSettings.aoColumns[i].bVisible )
					{
						if ( oSettings.aoColumns[i].sClass !== null )
						{
							nTh.className = oSettings.aoColumns[i].sClass;
						}
						
						if ( oSettings.aoColumns[i].sWidth !== null )
						{
							nTh.style.width = oSettings.aoColumns[i].sWidth;
						}
						
						nTr.appendChild( nTh );
					}
				}
				$('thead:eq(0)', oSettings.nTable).html( '' )[0].appendChild( nTr );
			}
			
			/* Add the extra markup needed by jQuery UI's themes */
			if ( oSettings.bJUI )
			{
				for ( i=0, iLen=oSettings.aoColumns.length ; i<iLen ; i++ )
				{
					oSettings.aoColumns[i].nTh.insertBefore( document.createElement('span'),
						oSettings.aoColumns[i].nTh.firstChild );
				}
			}
			
			/* Add sort listener */
			if ( oSettings.oFeatures.bSort )
			{
				for ( i=0 ; i<oSettings.aoColumns.length ; i++ )
				{
					if ( oSettings.aoColumns[i].bSortable !== false )
					{
						_fnSortAttachListener( oSettings, oSettings.aoColumns[i].nTh, i );
					}
					else
					{
						$(oSettings.aoColumns[i].nTh).addClass( oSettings.oClasses.sSortableNone );
					}
				}
				
				/* Take the brutal approach to cancelling text selection due to the shift key */
				$('thead:eq(0) th', oSettings.nTable).mousedown( function (e) {
					if ( e.shiftKey )
					{
						this.onselectstart = function() { return false; };
						return false;
					}
				} );
			}
			
			/* Cache the footer elements */
			var nTfoot = oSettings.nTable.getElementsByTagName('tfoot');
			if ( nTfoot.length !== 0 )
			{
				iCorrector = 0;
				var nTfs = nTfoot[0].getElementsByTagName('th');
				for ( i=0, iLen=nTfs.length ; i<iLen ; i++ )
				{
					oSettings.aoColumns[i].nTf = nTfs[i-iCorrector];
					if ( !oSettings.aoColumns[i].bVisible )
					{
						nTfs[i-iCorrector].parentNode.removeChild( nTfs[i-iCorrector] );
						iCorrector++;
					}
				}
			}
		}
		
		/*
		 * Function: _fnDraw
		 * Purpose:  Insert the required TR nodes into the table for display
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnDraw( oSettings )
		{
			var i, iLen;
			var anRows = [];
			var iRowCount = 0;
			var bRowError = false;
			var iStrips = oSettings.asStripClasses.length;
			var iOpenRows = oSettings.aoOpenRows.length;
			
			/* If we are dealing with Ajax - do it here */
			if ( oSettings.oFeatures.bServerSide && 
			     !_fnAjaxUpdate( oSettings ) )
			{
				return;
			}
			
			/* Check and see if we have an initial draw position from state saving */
			if ( typeof oSettings.iInitDisplayStart != 'undefined' && oSettings.iInitDisplayStart != -1 )
			{
				oSettings._iDisplayStart = (oSettings.iInitDisplayStart >= oSettings.fnRecordsDisplay()) ?
					0 : oSettings.iInitDisplayStart;
				oSettings.iInitDisplayStart = -1;
				_fnCalculateEnd( oSettings );
			}
			
			if ( oSettings.aiDisplay.length !== 0 )
			{
				var iStart = oSettings._iDisplayStart;
				var iEnd = oSettings._iDisplayEnd;
				
				if ( oSettings.oFeatures.bServerSide )
				{
					iStart = 0;
					iEnd = oSettings.aoData.length;
				}
				
				for ( var j=iStart ; j<iEnd ; j++ )
				{
					var aoData = oSettings.aoData[ oSettings.aiDisplay[j] ];
					var nRow = aoData.nTr;
					
					/* Remove the old stripping classes and then add the new one */
					if ( iStrips !== 0 )
					{
						var sStrip = oSettings.asStripClasses[ iRowCount % iStrips ];
						if ( aoData._sRowStripe != sStrip )
						{
							$(nRow).removeClass( aoData._sRowStripe ).addClass( sStrip );
							aoData._sRowStripe = sStrip;
						}
					}
					
					/* Custom row callback function - might want to manipule the row */
					if ( typeof oSettings.fnRowCallback == "function" )
					{
						nRow = oSettings.fnRowCallback( nRow, 
							oSettings.aoData[ oSettings.aiDisplay[j] ]._aData, iRowCount, j );
						if ( !nRow && !bRowError )
						{
							alert( "DataTables warning: A node was not returned by fnRowCallback" );
							bRowError = true;
						}
					}
					
					anRows.push( nRow );
					iRowCount++;
					
					/* If there is an open row - and it is attached to this parent - attach it on redraw */
					if ( iOpenRows !== 0 )
					{
						for ( var k=0 ; k<iOpenRows ; k++ )
						{
							if ( nRow == oSettings.aoOpenRows[k].nParent )
							{
								anRows.push( oSettings.aoOpenRows[k].nTr );
							}
						}
					}
				}
			}
			else
			{
				/* Table is empty - create a row with an empty message in it */
				anRows[ 0 ] = document.createElement( 'tr' );
				
				if ( typeof oSettings.asStripClasses[0] != 'undefined' )
				{
					anRows[ 0 ].className = oSettings.asStripClasses[0];
				}
				
				var nTd = document.createElement( 'td' );
				nTd.setAttribute( 'valign', "top" );
				nTd.colSpan = oSettings.aoColumns.length;
				nTd.className = oSettings.oClasses.sRowEmpty;
				nTd.innerHTML = oSettings.oLanguage.sZeroRecords;
				
				anRows[ iRowCount ].appendChild( nTd );
			}
			
			/* Callback the header and footer custom funcation if there is one */
			if ( typeof oSettings.fnHeaderCallback == 'function' )
			{
				oSettings.fnHeaderCallback( $('thead:eq(0)>tr', oSettings.nTable)[0], 
					_fnGetDataMaster( oSettings ), oSettings._iDisplayStart, oSettings.fnDisplayEnd(),
					oSettings.aiDisplay );
			}
			
			if ( typeof oSettings.fnFooterCallback == 'function' )
			{
				oSettings.fnFooterCallback( $('tfoot:eq(0)>tr', oSettings.nTable)[0], 
					_fnGetDataMaster( oSettings ), oSettings._iDisplayStart, oSettings.fnDisplayEnd(),
					oSettings.aiDisplay );
			}
			
			/* 
			 * Need to remove any old row from the display - note we can't just empty the tbody using
			 * $().html('') since this will unbind the jQuery event handlers (even although the node 
			 * still exists!) - equally we can't use innerHTML, since IE throws an exception.
			 */
			var nBody = oSettings.nTable.getElementsByTagName('tbody');
			if ( nBody[0] )
			{
				var nTrs = nBody[0].childNodes;
				for ( i=nTrs.length-1 ; i>=0 ; i-- )
				{
					nTrs[i].parentNode.removeChild( nTrs[i] );
				}
				
				/* Put the draw table into the dom */
				for ( i=0, iLen=anRows.length ; i<iLen ; i++ )
				{
					nBody[0].appendChild( anRows[i] );
				}
			}
			
			/* Call all required callback functions for the end of a draw */
			for ( i=0, iLen=oSettings.aoDrawCallback.length ; i<iLen ; i++ )
			{
				oSettings.aoDrawCallback[i].fn( oSettings );
			}
			
			/* Draw is complete, sorting and filtering must be as well */
			oSettings.bSorted = false;
			oSettings.bFiltered = false;
			
			/* Perform certain DOM operations after the table has been drawn for the first time */
			if ( typeof oSettings._bInitComplete == "undefined" )
			{
				oSettings._bInitComplete = true;
				
				/* Set an absolute width for the table such that pagination doesn't
				 * cause the table to resize
				 */
				if ( oSettings.oFeatures.bAutoWidth && oSettings.nTable.offsetWidth !== 0 )
				{
					oSettings.nTable.style.width = oSettings.nTable.offsetWidth+"px";
				}
			}
		}
		
		/*
		 * Function: _fnReDraw
		 * Purpose:  Redraw the table - taking account of the various features which are enabled
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnReDraw( oSettings )
		{
			if ( oSettings.oFeatures.bSort )
			{
				/* Sorting will refilter and draw for us */
				_fnSort( oSettings, oSettings.oPreviousSearch );
			}
			else if ( oSettings.oFeatures.bFilter )
			{
				/* Filtering will redraw for us */
				_fnFilterComplete( oSettings, oSettings.oPreviousSearch );
			}
			else
			{
				_fnCalculateEnd( oSettings );
				_fnDraw( oSettings );
			}
		}
		
		/*
		 * Function: _fnAjaxUpdate
		 * Purpose:  Update the table using an Ajax call
		 * Returns:  bool: block the table drawing or not
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnAjaxUpdate( oSettings )
		{
			if ( oSettings.bAjaxDataGet )
			{
				_fnProcessingDisplay( oSettings, true );
				var iColumns = oSettings.aoColumns.length;
				var aoData = [];
				var i;
				
				/* Paging and general */
				oSettings.iServerDraw++;
				aoData.push( { "name": "sEcho",          "value": oSettings.iServerDraw } );
				aoData.push( { "name": "iColumns",       "value": iColumns } );
				aoData.push( { "name": "sColumns",       "value": _fnColumnOrdering(oSettings) } );
				aoData.push( { "name": "iDisplayStart",  "value": oSettings._iDisplayStart } );
				aoData.push( { "name": "iDisplayLength", "value": oSettings.oFeatures.bPaginate !== false ?
					oSettings._iDisplayLength : -1 } );
				
				/* Filtering */
				if ( oSettings.oFeatures.bFilter !== false )
				{
					aoData.push( { "name": "sSearch",        "value": oSettings.oPreviousSearch.sSearch } );
					aoData.push( { "name": "bEscapeRegex",   "value": oSettings.oPreviousSearch.bEscapeRegex } );
					for ( i=0 ; i<iColumns ; i++ )
					{
						aoData.push( { "name": "sSearch_"+i,      "value": oSettings.aoPreSearchCols[i].sSearch } );
						aoData.push( { "name": "bEscapeRegex_"+i, "value": oSettings.aoPreSearchCols[i].bEscapeRegex } );
						aoData.push( { "name": "bSearchable_"+i,  "value": oSettings.aoColumns[i].bSearchable } );
					}
				}
				
				/* Sorting */
				if ( oSettings.oFeatures.bSort !== false )
				{
					var iFixed = oSettings.aaSortingFixed !== null ? oSettings.aaSortingFixed.length : 0;
					var iUser = oSettings.aaSorting.length;
					aoData.push( { "name": "iSortingCols",   "value": iFixed+iUser } );
					for ( i=0 ; i<iFixed ; i++ )
					{
						aoData.push( { "name": "iSortCol_"+i,  "value": oSettings.aaSortingFixed[i][0] } );
						aoData.push( { "name": "sSortDir_"+i,  "value": oSettings.aaSortingFixed[i][1] } );
					}
					
					for ( i=0 ; i<iUser ; i++ )
					{
						aoData.push( { "name": "iSortCol_"+(i+iFixed),  "value": oSettings.aaSorting[i][0] } );
						aoData.push( { "name": "sSortDir_"+(i+iFixed),  "value": oSettings.aaSorting[i][1] } );
					}
					
					for ( i=0 ; i<iColumns ; i++ )
					{
						aoData.push( { "name": "bSortable_"+i,  "value": oSettings.aoColumns[i].bSortable } );
					}
				}
				
				oSettings.fnServerData( oSettings.sAjaxSource, aoData, function(json) {
					_fnAjaxUpdateDraw( oSettings, json );
				} );
				return false;
			}
			else
			{
				return true;
			}
		}
		
		/*
		 * Function: _fnAjaxUpdateDraw
		 * Purpose:  Data the data from the server (nuking the old) and redraw the table
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 *           object:json - json data return from the server.
		 *             The following must be defined:
		 *               iTotalRecords, iTotalDisplayRecords, aaData
		 *             The following may be defined:
		 *               sColumns
		 */
		function _fnAjaxUpdateDraw ( oSettings, json )
		{
			if ( typeof json.sEcho != 'undefined' )
			{
				/* Protect against old returns over-writing a new one. Possible when you get
				 * very fast interaction, and later queires are completed much faster
				 */
				if ( json.sEcho*1 < oSettings.iServerDraw )
				{
					return;
				}
				else
				{
					oSettings.iServerDraw = json.sEcho * 1;
				}
			}
			
			_fnClearTable( oSettings );
			oSettings._iRecordsTotal = json.iTotalRecords;
			oSettings._iRecordsDisplay = json.iTotalDisplayRecords;
			
			/* Determine if reordering is required */
			var sOrdering = _fnColumnOrdering(oSettings);
			var bReOrder = (typeof json.sColumns != 'undefined' && sOrdering !== "" && json.sColumns != sOrdering );
			if ( bReOrder )
			{
				var aiIndex = _fnReOrderIndex( oSettings, json.sColumns );
			}
			
			for ( var i=0, iLen=json.aaData.length ; i<iLen ; i++ )
			{
				if ( bReOrder )
				{
					/* If we need to re-order, then create a new array with the correct order and add it */
					var aData = [];
					for ( var j=0, jLen=oSettings.aoColumns.length ; j<jLen ; j++ )
					{
						aData.push( json.aaData[i][ aiIndex[j] ] );
					}
					_fnAddData( oSettings, aData );
				}
				else
				{
					/* No re-order required, sever got it "right" - just straight add */
					_fnAddData( oSettings, json.aaData[i] );
				}
			}
			oSettings.aiDisplay = oSettings.aiDisplayMaster.slice();
			
			oSettings.bAjaxDataGet = false;
			_fnDraw( oSettings );
			oSettings.bAjaxDataGet = true;
			_fnProcessingDisplay( oSettings, false );
		}
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Options (features) HTML
		 */
		
		/*
		 * Function: _fnAddOptionsHtml
		 * Purpose:  Add the options to the page HTML for the table
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnAddOptionsHtml ( oSettings )
		{
			/*
			 * Create a temporary, empty, div which we can later on replace with what we have generated
			 * we do it this way to rendering the 'options' html offline - speed :-)
			 */
			var nHolding = document.createElement( 'div' );
			oSettings.nTable.parentNode.insertBefore( nHolding, oSettings.nTable );
			
			/* 
			 * All DataTables are wrapped in a div - this is not currently optional - backwards 
			 * compatability. It can be removed if you don't want it.
			 */
			var nWrapper = document.createElement( 'div' );
			nWrapper.className = oSettings.oClasses.sWrapper;
			if ( oSettings.sTableId !== '' )
			{
				nWrapper.setAttribute( 'id', oSettings.sTableId+'_wrapper' );
			}
			
			/* Track where we want to insert the option */
			var nInsertNode = nWrapper;
			
			/* Substitute any constants in the dom string */
			var sDom = oSettings.sDom.replace( "H", "fg-toolbar ui-widget-header ui-corner-tl ui-corner-tr ui-helper-clearfix" );
			sDom = sDom.replace( "F", "fg-toolbar ui-widget-header ui-corner-bl ui-corner-br ui-helper-clearfix" );
			
			/* Loop over the user set positioning and place the elements as needed */
			var aDom = sDom.split('');
			var nTmp, iPushFeature, cOption, nNewNode, cNext, sClass, j;
			for ( var i=0 ; i<aDom.length ; i++ )
			{
				iPushFeature = 0;
				cOption = aDom[i];
				
				if ( cOption == '<' )
				{
					/* New container div */
					nNewNode = document.createElement( 'div' );
					
					/* Check to see if we should append a class name to the container */
					cNext = aDom[i+1];
					if ( cNext == "'" || cNext == '"' )
					{
						sClass = "";
						j = 2;
						while ( aDom[i+j] != cNext )
						{
							sClass += aDom[i+j];
							j++;
						}
						nNewNode.className = sClass;
						i += j; /* Move along the position array */
					}
					
					nInsertNode.appendChild( nNewNode );
					nInsertNode = nNewNode;
				}
				else if ( cOption == '>' )
				{
					/* End container div */
					nInsertNode = nInsertNode.parentNode;
				}
				else if ( cOption == 'l' && oSettings.oFeatures.bPaginate && oSettings.oFeatures.bLengthChange )
				{
					/* Length */
					nTmp = _fnFeatureHtmlLength( oSettings );
					iPushFeature = 1;
				}
				else if ( cOption == 'f' && oSettings.oFeatures.bFilter )
				{
					/* Filter */
					nTmp = _fnFeatureHtmlFilter( oSettings );
					iPushFeature = 1;
				}
				else if ( cOption == 'r' && oSettings.oFeatures.bProcessing )
				{
					/* pRocessing */
					nTmp = _fnFeatureHtmlProcessing( oSettings );
					iPushFeature = 1;
				}
				else if ( cOption == 't' )
				{
					/* Table */
					nTmp = oSettings.nTable;
					iPushFeature = 1;
				}
				else if ( cOption ==  'i' && oSettings.oFeatures.bInfo )
				{
					/* Info */
					nTmp = _fnFeatureHtmlInfo( oSettings );
					iPushFeature = 1;
				}
				else if ( cOption == 'p' && oSettings.oFeatures.bPaginate )
				{
					/* Pagination */
					nTmp = _fnFeatureHtmlPaginate( oSettings );
					iPushFeature = 1;
				}
				else if ( _oExt.aoFeatures.length !== 0 )
				{
					/* Plug-in features */
					var aoFeatures = _oExt.aoFeatures;
					for ( var k=0, kLen=aoFeatures.length ; k<kLen ; k++ )
					{
						if ( cOption == aoFeatures[k].cFeature )
						{
							nTmp = aoFeatures[k].fnInit( oSettings );
							if ( nTmp )
							{
								iPushFeature = 1;
							}
							break;
						}
					}
				}
				
				/* Add to the 2D features array */
				if ( iPushFeature == 1 )
				{
					if ( typeof oSettings.aanFeatures[cOption] != 'object' )
					{
						oSettings.aanFeatures[cOption] = [];
					}
					oSettings.aanFeatures[cOption].push( nTmp );
					nInsertNode.appendChild( nTmp );
				}
			}
			
			/* Built our DOM structure - replace the holding div with what we want */
			nHolding.parentNode.replaceChild( nWrapper, nHolding );
		}
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Feature: Filtering
		 */
		
		/*
		 * Function: _fnFeatureHtmlFilter
		 * Purpose:  Generate the node required for filtering text
		 * Returns:  node
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnFeatureHtmlFilter ( oSettings )
		{
			var nFilter = document.createElement( 'div' );
			if ( oSettings.sTableId !== '' && typeof oSettings.aanFeatures.f == "undefined" )
			{
				nFilter.setAttribute( 'id', oSettings.sTableId+'_filter' );
			}
			nFilter.className = oSettings.oClasses.sFilter;
			var sSpace = oSettings.oLanguage.sSearch==="" ? "" : " ";
			nFilter.innerHTML = oSettings.oLanguage.sSearch+sSpace+'<input type="text" />';
			
			var jqFilter = $("input", nFilter);
			jqFilter.val( oSettings.oPreviousSearch.sSearch.replace('"','&quot;') );
			jqFilter.keyup( function(e) {
				/* Update all other filter input elements for the new display */
				var n = oSettings.aanFeatures.f;
				for ( var i=0, iLen=n.length ; i<iLen ; i++ )
				{
					if ( n[i] != this.parentNode )
					{
						$('input', n[i]).val( this.value );
					}
				}
				
				/* Now do the filter */
				_fnFilterComplete( oSettings, { 
					"sSearch": this.value, 
					"bEscapeRegex": oSettings.oPreviousSearch.bEscapeRegex 
				} );
			} );
			
			jqFilter.keypress( function(e) {
				/* Prevent default */
				if ( e.keyCode == 13 )
				{
					return false;
				}
			} );
			
			return nFilter;
		}
		
		/*
		 * Function: _fnFilterComplete
		 * Purpose:  Filter the table using both the global filter and column based filtering
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 *           object:oSearch: search information
		 *           int:iForce - optional - force a research of the master array (1) or not (undefined or 0)
		 */
		function _fnFilterComplete ( oSettings, oInput, iForce )
		{
			/* Filter on everything */
			_fnFilter( oSettings, oInput.sSearch, iForce, oInput.bEscapeRegex );
			
			/* Now do the individual column filter */
			for ( var i=0 ; i<oSettings.aoPreSearchCols.length ; i++ )
			{
				_fnFilterColumn( oSettings, oSettings.aoPreSearchCols[i].sSearch, i, 
					oSettings.aoPreSearchCols[i].bEscapeRegex );
			}
			
			/* Custom filtering */
			if ( _oExt.afnFiltering.length !== 0 )
			{
				_fnFilterCustom( oSettings );
			}
			
			/* Tell the draw function we have been filtering */
			oSettings.bFiltered = true;
			
			/* Redraw the table */
			oSettings._iDisplayStart = 0;
			_fnCalculateEnd( oSettings );
			_fnDraw( oSettings );
			
			/* Rebuild search array 'offline' */
			_fnBuildSearchArray( oSettings, 0 );
		}
		
		/*
		 * Function: _fnFilterCustom
		 * Purpose:  Apply custom filtering functions
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnFilterCustom( oSettings )
		{
			var afnFilters = _oExt.afnFiltering;
			for ( var i=0, iLen=afnFilters.length ; i<iLen ; i++ )
			{
				var iCorrector = 0;
				for ( var j=0, jLen=oSettings.aiDisplay.length ; j<jLen ; j++ )
				{
					var iDisIndex = oSettings.aiDisplay[j-iCorrector];
					
					/* Check if we should use this row based on the filtering function */
					if ( !afnFilters[i]( oSettings, oSettings.aoData[iDisIndex]._aData, iDisIndex ) )
					{
						oSettings.aiDisplay.splice( j-iCorrector, 1 );
						iCorrector++;
					}
				}
			}
		}
		
		/*
		 * Function: _fnFilterColumn
		 * Purpose:  Filter the table on a per-column basis
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 *           string:sInput - string to filter on
		 *           int:iColumn - column to filter
		 *           bool:bEscapeRegex - escape regex or not
		 */
		function _fnFilterColumn ( oSettings, sInput, iColumn, bEscapeRegex )
		{
			if ( sInput === "" )
			{
				return;
			}
			
			var iIndexCorrector = 0;
			var sRegexMatch = bEscapeRegex ? _fnEscapeRegex( sInput ) : sInput;
			var rpSearch = new RegExp( sRegexMatch, "i" );
			
			for ( var i=oSettings.aiDisplay.length-1 ; i>=0 ; i-- )
			{
				var sData = _fnDataToSearch( oSettings.aoData[ oSettings.aiDisplay[i] ]._aData[iColumn],
					oSettings.aoColumns[iColumn].sType );
				if ( ! rpSearch.test( sData ) )
				{
					oSettings.aiDisplay.splice( i, 1 );
					iIndexCorrector++;
				}
			}
		}
		
		/*
		 * Function: _fnFilter
		 * Purpose:  Filter the data table based on user input and draw the table
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 *           string:sInput - string to filter on
		 *           int:iForce - optional - force a research of the master array (1) or not (undefined or 0)
		 *           bool:bEscapeRegex - escape regex or not
		 */
		function _fnFilter( oSettings, sInput, iForce, bEscapeRegex )
		{
			var i;
			
			/* Check if we are forcing or not - optional parameter */
			if ( typeof iForce == 'undefined' || iForce === null )
			{
				iForce = 0;
			}
			
			/* Need to take account of custom filtering functions always */
			if ( _oExt.afnFiltering.length !== 0 )
			{
				iForce = 1;
			}
			
			/* Generate the regular expression to use. Something along the lines of:
			 * ^(?=.*?\bone\b)(?=.*?\btwo\b)(?=.*?\bthree\b).*$
			 */
			var asSearch = bEscapeRegex ?
				_fnEscapeRegex( sInput ).split( ' ' ) :
				sInput.split( ' ' );
			var sRegExpString = '^(?=.*?'+asSearch.join( ')(?=.*?' )+').*$';
			var rpSearch = new RegExp( sRegExpString, "i" ); /* case insensitive */
			
			/*
			 * If the input is blank - we want the full data set
			 */
			if ( sInput.length <= 0 )
			{
				oSettings.aiDisplay.splice( 0, oSettings.aiDisplay.length);
				oSettings.aiDisplay = oSettings.aiDisplayMaster.slice();
			}
			else
			{
				/*
				 * We are starting a new search or the new search string is smaller 
				 * then the old one (i.e. delete). Search from the master array
			 	 */
				if ( oSettings.aiDisplay.length == oSettings.aiDisplayMaster.length ||
					   oSettings.oPreviousSearch.sSearch.length > sInput.length || iForce == 1 ||
					   sInput.indexOf(oSettings.oPreviousSearch.sSearch) !== 0 )
				{
					/* Nuke the old display array - we are going to rebuild it */
					oSettings.aiDisplay.splice( 0, oSettings.aiDisplay.length);
					
					/* Force a rebuild of the search array */
					_fnBuildSearchArray( oSettings, 1 );
					
					/* Search through all records to populate the search array
					 * The the oSettings.aiDisplayMaster and asDataSearch arrays have 1 to 1 
					 * mapping
					 */
					for ( i=0 ; i<oSettings.aiDisplayMaster.length ; i++ )
					{
						if ( rpSearch.test(oSettings.asDataSearch[i]) )
						{
							oSettings.aiDisplay.push( oSettings.aiDisplayMaster[i] );
						}
					}
			  }
			  else
				{
			  	/* Using old search array - refine it - do it this way for speed
			  	 * Don't have to search the whole master array again
			 		 */
			  	var iIndexCorrector = 0;
			  	
			  	/* Search the current results */
			  	for ( i=0 ; i<oSettings.asDataSearch.length ; i++ )
					{
			  		if ( ! rpSearch.test(oSettings.asDataSearch[i]) )
						{
			  			oSettings.aiDisplay.splice( i-iIndexCorrector, 1 );
			  			iIndexCorrector++;
			  		}
			  	}
			  }
			}
			oSettings.oPreviousSearch.sSearch = sInput;
			oSettings.oPreviousSearch.bEscapeRegex = bEscapeRegex;
		}
		
		/*
		 * Function: _fnBuildSearchArray
		 * Purpose:  Create an array which can be quickly search through
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 *           int:iMaster - use the master data array - optional
		 */
		function _fnBuildSearchArray ( oSettings, iMaster )
		{
			/* Clear out the old data */
			oSettings.asDataSearch.splice( 0, oSettings.asDataSearch.length );
			
			var aArray = (typeof iMaster != 'undefined' && iMaster == 1) ?
			 	oSettings.aiDisplayMaster : oSettings.aiDisplay;
			
			for ( var i=0, iLen=aArray.length ; i<iLen ; i++ )
			{
				oSettings.asDataSearch[i] = '';
				for ( var j=0, jLen=oSettings.aoColumns.length ; j<jLen ; j++ )
				{
					if ( oSettings.aoColumns[j].bSearchable )
					{
						var sData = oSettings.aoData[ aArray[i] ]._aData[j];
						oSettings.asDataSearch[i] += _fnDataToSearch( sData, oSettings.aoColumns[j].sType )+' ';
					}
				}
			}
		}
		
		/*
		 * Function: _fnDataToSearch
		 * Purpose:  Convert raw data into something that the user can search on
		 * Returns:  string: - search string
		 * Inputs:   string:sData - data to be modified
		 *           string:sType - data type
		 */
		function _fnDataToSearch ( sData, sType )
		{
			
			if ( typeof _oExt.ofnSearch[sType] == "function" )
			{
				return _oExt.ofnSearch[sType]( sData );
			}
			else if ( sType == "html" )
			{
				return sData.replace(/\n/g," ").replace( /<.*?>/g, "" );
			}
			else if ( typeof sData == "string" )
			{
				return sData.replace(/\n/g," ");
			}
			return sData;
		}
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Feature: Sorting
		 */
		
		/*
	 	 * Function: _fnSort
		 * Purpose:  Change the order of the table
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 *           bool:bApplyClasses - optional - should we apply classes or not
		 * Notes:    We always sort the master array and then apply a filter again
		 *   if it is needed. This probably isn't optimal - but atm I can't think
		 *   of any other way which is (each has disadvantages). we want to sort aiDisplayMaster - 
		 *   but according to aoData[]._aData
		 */
		function _fnSort ( oSettings, bApplyClasses )
		{
			var aaSort = [];
			var oSort = _oExt.oSort;
			var aoData = oSettings.aoData;
			var iDataSort;
			var iDataType;
			var i, j, jLen;
			
			/* No sorting required if server-side or no sorting array */
			if ( !oSettings.oFeatures.bServerSide && 
				(oSettings.aaSorting.length !== 0 || oSettings.aaSortingFixed !== null) )
			{
				if ( oSettings.aaSortingFixed !== null )
				{
					aaSort = oSettings.aaSortingFixed.concat( oSettings.aaSorting );
				}
				else
				{
					aaSort = oSettings.aaSorting.slice();
				}
				
				/* If there is a sorting data type, and a fuction belonging to it, then we need to
				 * get the data from the developer's function and apply it for this column
				 */
				for ( i=0 ; i<aaSort.length ; i++ )
				{
					var iColumn = aaSort[i][0];
					var sDataType = oSettings.aoColumns[ iColumn ].sSortDataType;
					if ( typeof _oExt.afnSortData[sDataType] != 'undefined' )
					{
						var iCorrector = 0;
						var aData = _oExt.afnSortData[sDataType]( oSettings, iColumn );
						for ( j=0, jLen=aoData.length ; j<jLen ; j++ )
						{
							if ( aoData[j] !== null )
							{
								aoData[j]._aData[iColumn] = aData[iCorrector];
								iCorrector++;
							}
						}
					}
				}
				
				/* DataTables offers two different methods for doing the 2D array sorting over multiple
				 * columns. The first is to construct a function dynamically, and then evaluate and run
				 * the function, while the second has no need for evalulation, but is a little bit slower.
				 * This is used for environments which do not allow eval() for code execuation such as AIR
				 */
				if ( !window.runtime )
				{
					/* Dynamically created sorting function. Based on the information that we have, we can
					 * create a sorting function as if it were specifically written for this sort. Here we
					 * want to build a function something like (for two column sorting):
					 *  fnLocalSorting = function(a,b){
					 *  	var iTest;
					 *  	iTest = oSort['string-asc']('data11', 'data12');
					 *  	if (iTest === 0)
					 *  		iTest = oSort['numeric-desc']('data21', 'data22');
					 *  		if (iTest === 0)
					 *  			return oSort['numeric-desc'](1,2);
					 *  	return iTest;
					 *  }
					 * So basically we have a test for each column, and if that column matches, test the
					 * next one. If all columns match, then we use a numeric sort on the position the two
					 * row have in the original data array in order to provide a stable sort.
					 */
					var fnLocalSorting;
					var sDynamicSort = "fnLocalSorting = function(a,b){"+
						"var iTest;";
					
					for ( i=0 ; i<aaSort.length-1 ; i++ )
					{
						iDataSort = oSettings.aoColumns[ aaSort[i][0] ].iDataSort;
						iDataType = oSettings.aoColumns[ iDataSort ].sType;
						sDynamicSort += "iTest = oSort['"+iDataType+"-"+aaSort[i][1]+"']"+
							"( aoData[a]._aData["+iDataSort+"], aoData[b]._aData["+iDataSort+"] ); if ( iTest === 0 )";
					}
					
					iDataSort = oSettings.aoColumns[ aaSort[aaSort.length-1][0] ].iDataSort;
					iDataType = oSettings.aoColumns[ iDataSort ].sType;
					sDynamicSort += "iTest = oSort['"+iDataType+"-"+aaSort[aaSort.length-1][1]+"']"+
						"( aoData[a]._aData["+iDataSort+"], aoData[b]._aData["+iDataSort+"] );"+
						"if (iTest===0) return oSort['numeric-"+aaSort[aaSort.length-1][1]+"'](a, b); "+
						"return iTest;}";
					
					/* The eval has to be done to a variable for IE */
					eval( sDynamicSort );
					oSettings.aiDisplayMaster.sort( fnLocalSorting );
				}
				else
				{
					/*
					 * Non-eval() sorting (AIR and other environments which doesn't allow code in eval()
					 * Note that for reasonable sized data sets this method is around 1.5 times slower than
					 * the eval above (hence why it is not used all the time). Oddly enough, it is ever so
					 * slightly faster for very small sets (presumably the eval has overhead).
					 *   Single column (1083 records) - eval: 32mS   AIR: 38mS
					 *   Two columns (1083 records) -   eval: 55mS   AIR: 66mS
					 */
					
					/* Build a cached array so the sort doesn't have to process this stuff on every call */
					var aAirSort = [];
					var iLen = aaSort.length;
					for ( i=0 ; i<iLen ; i++ )
					{
						iDataSort = oSettings.aoColumns[ aaSort[i][0] ].iDataSort;
						aAirSort.push( [
							iDataSort,
							oSettings.aoColumns[ iDataSort ].sType+'-'+aaSort[i][1]
						] );
					}
					
					oSettings.aiDisplayMaster.sort( function (a,b) {
						var iTest;
						for ( var i=0 ; i<iLen ; i++ )
						{
							iTest = oSort[ aAirSort[i][1] ]( aoData[a]._aData[aAirSort[i][0]], aoData[b]._aData[aAirSort[i][0]] );
							if ( iTest !== 0 )
							{
								return iTest;
							}
						}
						return 0;
					} );
				}
			}
			
			/* Alter the sorting classes to take account of the changes */
			if ( typeof bApplyClasses == 'undefined' || bApplyClasses )
			{
				_fnSortingClasses( oSettings );
			}
			
			/* Tell the draw function that we have sorted the data */
			oSettings.bSorted = true;
			
			/* Copy the master data into the draw array and re-draw */
			if ( oSettings.oFeatures.bFilter )
			{
				/* _fnFilter() will redraw the table for us */
				_fnFilterComplete( oSettings, oSettings.oPreviousSearch, 1 );
			}
			else
			{
				oSettings.aiDisplay = oSettings.aiDisplayMaster.slice();
				oSettings._iDisplayStart = 0; /* reset display back to page 0 */
				_fnCalculateEnd( oSettings );
				_fnDraw( oSettings );
			}
		}
		
		/*
		 * Function: _fnSortAttachListener
		 * Purpose:  Attach a sort handler (click) to a node
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 *           node:nNode - node to attach the handler to
		 *           int:iDataIndex - column sorting index
		 *           function:fnCallback - callback function - optional
		 */
		function _fnSortAttachListener ( oSettings, nNode, iDataIndex, fnCallback )
		{
			$(nNode).click( function (e) {
				/* If the column is not sortable - don't to anything */
				if ( oSettings.aoColumns[iDataIndex].bSortable === false )
				{
					return;
				}
				
				/*
				 * This is a little bit odd I admit... I declare a temporary function inside the scope of
				 * _fnDrawHead and the click handler in order that the code presented here can be used 
				 * twice - once for when bProcessing is enabled, and another time for when it is 
				 * disabled, as we need to perform slightly different actions.
				 *   Basically the issue here is that the Javascript engine in modern browsers don't 
				 * appear to allow the rendering engine to update the display while it is still excuting
				 * it's thread (well - it does but only after long intervals). This means that the 
				 * 'processing' display doesn't appear for a table sort. To break the js thread up a bit
				 * I force an execution break by using setTimeout - but this breaks the expected 
				 * thread continuation for the end-developer's point of view (their code would execute
				 * too early), so we on;y do it when we absolutely have to.
				 */
				var fnInnerSorting = function () {
					var iColumn, iNextSort;
					
					/* If the shift key is pressed then we are multipe column sorting */
					if ( e.shiftKey )
					{
						/* Are we already doing some kind of sort on this column? */
						var bFound = false;
						for ( var i=0 ; i<oSettings.aaSorting.length ; i++ )
						{
							if ( oSettings.aaSorting[i][0] == iDataIndex )
							{
								bFound = true;
								iColumn = oSettings.aaSorting[i][0];
								iNextSort = oSettings.aaSorting[i][2]+1;
								
								if ( typeof oSettings.aoColumns[iColumn].asSorting[iNextSort] == 'undefined' )
								{
									/* Reached the end of the sorting options, remove from multi-col sort */
									oSettings.aaSorting.splice( i, 1 );
								}
								else
								{
									/* Move onto next sorting direction */
									oSettings.aaSorting[i][1] = oSettings.aoColumns[iColumn].asSorting[iNextSort];
									oSettings.aaSorting[i][2] = iNextSort;
								}
								break;
							}
						}
						
						/* No sort yet - add it in */
						if ( bFound === false )
						{
							oSettings.aaSorting.push( [ iDataIndex, 
								oSettings.aoColumns[iDataIndex].asSorting[0], 0 ] );
						}
					}
					else
					{
						/* If no shift key then single column sort */
						if ( oSettings.aaSorting.length == 1 && oSettings.aaSorting[0][0] == iDataIndex )
						{
							iColumn = oSettings.aaSorting[0][0];
							iNextSort = oSettings.aaSorting[0][2]+1;
							if ( typeof oSettings.aoColumns[iColumn].asSorting[iNextSort] == 'undefined' )
							{
								iNextSort = 0;
							}
							oSettings.aaSorting[0][1] = oSettings.aoColumns[iColumn].asSorting[iNextSort];
							oSettings.aaSorting[0][2] = iNextSort;
						}
						else
						{
							oSettings.aaSorting.splice( 0, oSettings.aaSorting.length );
							oSettings.aaSorting.push( [ iDataIndex, 
								oSettings.aoColumns[iDataIndex].asSorting[0], 0 ] );
						}
					}
					
					/* Run the sort */
					_fnSort( oSettings );
				}; /* /fnInnerSorting */
				
				if ( !oSettings.oFeatures.bProcessing )
				{
					fnInnerSorting();
				}
				else
				{
					_fnProcessingDisplay( oSettings, true );
					setTimeout( function() {
						fnInnerSorting();
						if ( !oSettings.oFeatures.bServerSide )
						{
							_fnProcessingDisplay( oSettings, false );
						}
					}, 0 );
				}
				
				/* Call the user specified callback function - used for async user interaction */
				if ( typeof fnCallback == 'function' )
				{
					fnCallback( oSettings );
				}
			} );
		}
		
		/*
		 * Function: _fnSortingClasses
		 * Purpose:  Set the sortting classes on the header
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 * Notes:    It is safe to call this function when bSort is false
		 */
		function _fnSortingClasses( oSettings )
		{
			var i, iLen, j, jLen, iFound;
			var aaSort, sClass;
			var iColumns = oSettings.aoColumns.length;
			var oClasses = oSettings.oClasses;
			
			for ( i=0 ; i<iColumns ; i++ )
			{
				if ( oSettings.aoColumns[i].bSortable )
				{
					$(oSettings.aoColumns[i].nTh).removeClass( oClasses.sSortAsc +" "+ oClasses.sSortDesc +
				 		" "+ oSettings.aoColumns[i].sSortingClass );
				}
			}
			
			if ( oSettings.aaSortingFixed !== null )
			{
				aaSort = oSettings.aaSortingFixed.concat( oSettings.aaSorting );
			}
			else
			{
				aaSort = oSettings.aaSorting.slice();
			}
			
			/* Apply the required classes to the header */
			for ( i=0 ; i<oSettings.aoColumns.length ; i++ )
			{
				if ( oSettings.aoColumns[i].bSortable )
				{
					sClass = oSettings.aoColumns[i].sSortingClass;
					iFound = -1;
					for ( j=0 ; j<aaSort.length ; j++ )
					{
						if ( aaSort[j][0] == i )
						{
							sClass = ( aaSort[j][1] == "asc" ) ?
								oClasses.sSortAsc : oClasses.sSortDesc;
							iFound = j;
							break;
						}
					}
					$(oSettings.aoColumns[i].nTh).addClass( sClass );
					
					if ( oSettings.bJUI )
					{
						/* jQuery UI uses extra markup */
						var jqSpan = $("span", oSettings.aoColumns[i].nTh);
						jqSpan.removeClass(oClasses.sSortJUIAsc +" "+ oClasses.sSortJUIDesc +" "+ 
							oClasses.sSortJUI +" "+ oClasses.sSortJUIAscAllowed +" "+ oClasses.sSortJUIDescAllowed );
						
						var sSpanClass;
						if ( iFound == -1 )
						{
						 	sSpanClass = oSettings.aoColumns[i].sSortingClassJUI;
						}
						else if ( aaSort[iFound][1] == "asc" )
						{
							sSpanClass = oClasses.sSortJUIAsc;
						}
						else
						{
							sSpanClass = oClasses.sSortJUIDesc;
						}
						
						jqSpan.addClass( sSpanClass );
					}
				}
				else
				{
					/* No sorting on this column, so add the base class. This will have been assigned by
					 * _fnAddColumn
					 */
					$(oSettings.aoColumns[i].nTh).addClass( oSettings.aoColumns[i].sSortingClass );
				}
			}
			
			/* 
			 * Apply the required classes to the table body
			 * Note that this is given as a feature switch since it can significantly slow down a sort
			 * on large data sets (adding and removing of classes is always slow at the best of times..)
			 * Further to this, note that this code is admitadly fairly ugly. It could be made a lot 
			 * simpiler using jQuery selectors and add/removeClass, but that is significantly slower
			 * (on the order of 5 times slower) - hence the direct DOM manipulation here.
			 */
			sClass = oClasses.sSortColumn;
			
			if ( oSettings.oFeatures.bSort && oSettings.oFeatures.bSortClasses )
			{
				var nTds = _fnGetTdNodes( oSettings );
				
				/* Remove the old classes */
				if ( nTds.length >= iColumns )
				{
					for ( i=0 ; i<iColumns ; i++ )
					{
						if ( nTds[i].className.indexOf(sClass+"1") != -1 )
						{
							for ( j=0, jLen=(nTds.length/iColumns) ; j<jLen ; j++ )
							{
								nTds[(iColumns*j)+i].className = 
									nTds[(iColumns*j)+i].className.replace( " "+sClass+"1", "" );
							}
						}
						else if ( nTds[i].className.indexOf(sClass+"2") != -1 )
						{
							for ( j=0, jLen=(nTds.length/iColumns) ; j<jLen ; j++ )
							{
								nTds[(iColumns*j)+i].className = 
									nTds[(iColumns*j)+i].className.replace( " "+sClass+"2", "" );
							}
						}
						else if ( nTds[i].className.indexOf(sClass+"3") != -1 )
						{
							for ( j=0, jLen=(nTds.length/iColumns) ; j<jLen ; j++ )
							{
								nTds[(iColumns*j)+i].className = 
									nTds[(iColumns*j)+i].className.replace( " "+sClass+"3", "" );
							}
						}
					}
				}
				
				/* Add the new classes to the table */
				var iClass = 1, iTargetCol;
				for ( i=0 ; i<aaSort.length ; i++ )
				{
					iTargetCol = parseInt( aaSort[i][0], 10 );
					for ( j=0, jLen=(nTds.length/iColumns) ; j<jLen ; j++ )
					{
						nTds[(iColumns*j)+iTargetCol].className += " "+sClass+iClass;
					}
					
					if ( iClass < 3 )
					{
						iClass++;
					}
				}
			}
		}
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Feature: Pagination. Note that most of the paging logic is done in 
		 * _oExt.oPagination
		 */
		
		/*
		 * Function: _fnFeatureHtmlPaginate
		 * Purpose:  Generate the node required for default pagination
		 * Returns:  node
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnFeatureHtmlPaginate ( oSettings )
		{
			var nPaginate = document.createElement( 'div' );
			nPaginate.className = oSettings.oClasses.sPaging+oSettings.sPaginationType;
			
			_oExt.oPagination[ oSettings.sPaginationType ].fnInit( oSettings, nPaginate, 
				function( oSettings ) {
					_fnCalculateEnd( oSettings );
					_fnDraw( oSettings );
				}
			);
			
			/* Add a draw callback for the pagination on first instance, to update the paging display */
			if ( typeof oSettings.aanFeatures.p == "undefined" )
			{
				oSettings.aoDrawCallback.push( {
					"fn": function( oSettings ) {
						_oExt.oPagination[ oSettings.sPaginationType ].fnUpdate( oSettings, function( oSettings ) {
							_fnCalculateEnd( oSettings );
							_fnDraw( oSettings );
						} );
					},
					"sName": "pagination"
				} );
			}
			return nPaginate;
		}
		
		/*
		 * Function: _fnPageChange
		 * Purpose:  Alter the display settings to change the page
		 * Returns:  bool:true - page has changed, false - no change (no effect) eg 'first' on page 1
		 * Inputs:   object:oSettings - dataTables settings object
		 *           string:sAction - paging action to take: "first", "previous", "next" or "last"
		 */
		function _fnPageChange ( oSettings, sAction )
		{
			var iOldStart = oSettings._iDisplayStart;
			
			if ( sAction == "first" )
			{
				oSettings._iDisplayStart = 0;
			}
			else if ( sAction == "previous" )
			{
				oSettings._iDisplayStart = oSettings._iDisplayLength>=0 ?
					oSettings._iDisplayStart - oSettings._iDisplayLength :
					0;
				
				/* Correct for underrun */
				if ( oSettings._iDisplayStart < 0 )
				{
				  oSettings._iDisplayStart = 0;
				}
			}
			else if ( sAction == "next" )
			{
				if ( oSettings._iDisplayLength >= 0 )
				{
					/* Make sure we are not over running the display array */
					if ( oSettings._iDisplayStart + oSettings._iDisplayLength < oSettings.fnRecordsDisplay() )
					{
						oSettings._iDisplayStart += oSettings._iDisplayLength;
					}
				}
				else
				{
					oSettings._iDisplayStart = 0;
				}
			}
			else if ( sAction == "last" )
			{
				if ( oSettings._iDisplayLength >= 0 )
				{
					var iPages = parseInt( (oSettings.fnRecordsDisplay()-1) / oSettings._iDisplayLength, 10 ) + 1;
					oSettings._iDisplayStart = (iPages-1) * oSettings._iDisplayLength;
				}
				else
				{
					oSettings._iDisplayStart = 0;
				}
			}
			else
			{
				alert( "DataTables warning: unknown paging action: "+sAction );
			}
			
			return iOldStart != oSettings._iDisplayStart;
		}
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Feature: HTML info
		 */
		
		/*
		 * Function: _fnFeatureHtmlInfo
		 * Purpose:  Generate the node required for the info display
		 * Returns:  node
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnFeatureHtmlInfo ( oSettings )
		{
			var nInfo = document.createElement( 'div' );
			nInfo.className = oSettings.oClasses.sInfo;
			
			/* Actions that are to be taken once only for this feature */
			if ( typeof oSettings.aanFeatures.i == "undefined" )
			{
				/* Add draw callback */
				oSettings.aoDrawCallback.push( {
					"fn": _fnUpdateInfo,
					"sName": "information"
				} );
				
				/* Add id */
				if ( oSettings.sTableId !== '' )
				{
					nInfo.setAttribute( 'id', oSettings.sTableId+'_info' );
				}
			}
			
			return nInfo;
		}
		
		/*
		 * Function: _fnUpdateInfo
		 * Purpose:  Update the information elements in the display
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnUpdateInfo ( oSettings )
		{
			/* Show information about the table */
			if ( !oSettings.oFeatures.bInfo || oSettings.aanFeatures.i.length === 0 )
			{
				return;
			}
			
			var nFirst = oSettings.aanFeatures.i[0];
			
			if ( oSettings.fnRecordsDisplay() === 0 && 
				   oSettings.fnRecordsDisplay() == oSettings.fnRecordsTotal() )
			{
				/* Empty record set */
				nFirst.innerHTML = oSettings.oLanguage.sInfoEmpty+ oSettings.oLanguage.sInfoPostFix;
			}
			else if ( oSettings.fnRecordsDisplay() === 0 )
			{
				/* Rmpty record set after filtering */
				nFirst.innerHTML = oSettings.oLanguage.sInfoEmpty +' '+ 
					oSettings.oLanguage.sInfoFiltered.replace('_MAX_', 
						oSettings.fnRecordsTotal())+ oSettings.oLanguage.sInfoPostFix;
			}
			else if ( oSettings.fnRecordsDisplay() == oSettings.fnRecordsTotal() )
			{
				/* Normal record set */
				nFirst.innerHTML = oSettings.oLanguage.sInfo.
						replace('_START_',oSettings._iDisplayStart+1).
						replace('_END_',oSettings.fnDisplayEnd()).
						replace('_TOTAL_',oSettings.fnRecordsDisplay())+ 
					oSettings.oLanguage.sInfoPostFix;
			}
			else
			{
				/* Record set after filtering */
				nFirst.innerHTML = 
					oSettings.oLanguage.sInfo.
						replace('_START_',oSettings._iDisplayStart+1).
						replace('_END_',oSettings.fnDisplayEnd()).
						replace('_TOTAL_',oSettings.fnRecordsDisplay()) +' '+ 
					oSettings.oLanguage.sInfoFiltered.replace('_MAX_', oSettings.fnRecordsTotal())+ 
					oSettings.oLanguage.sInfoPostFix;
			}
			
			/* No point in recalculating for the other info elements, just copy the first one in */
			var n = oSettings.aanFeatures.i;
			if ( n.length > 1 )
			{
				var sInfo = nFirst.innerHTML;
				for ( var i=1, iLen=n.length ; i<iLen ; i++ )
				{
					n[i].innerHTML = sInfo;
				}
			}
		}
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Feature: Length change
		 */
		
		/*
		 * Function: _fnFeatureHtmlLength
		 * Purpose:  Generate the node required for user display length changing
		 * Returns:  node
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnFeatureHtmlLength ( oSettings )
		{
			/* This can be overruled by not using the _MENU_ var/macro in the language variable */
			var sName = (oSettings.sTableId === "") ? "" : 'name="'+oSettings.sTableId+'_length"';
			var sStdMenu = 
				'<select size="1" '+sName+'>'+
					'<option value="10">10</option>'+
					'<option value="25">25</option>'+
					'<option value="50">50</option>'+
					'<option value="100">100</option>'+
				'</select>';
			
			var nLength = document.createElement( 'div' );
			if ( oSettings.sTableId !== '' && typeof oSettings.aanFeatures.l == "undefined" )
			{
				nLength.setAttribute( 'id', oSettings.sTableId+'_length' );
			}
			nLength.className = oSettings.oClasses.sLength;
			nLength.innerHTML = oSettings.oLanguage.sLengthMenu.replace( '_MENU_', sStdMenu );
			
			/*
			 * Set the length to the current display length - thanks to Andrea Pavlovic for this fix,
			 * and Stefan Skopnik for fixing the fix!
			 */
			$('select option[value="'+oSettings._iDisplayLength+'"]',nLength).attr("selected",true);
			
			$('select', nLength).change( function(e) {
				var iVal = $(this).val();
				
				/* Update all other length options for the new display */
				var n = oSettings.aanFeatures.l;
				for ( var i=0, iLen=n.length ; i<iLen ; i++ )
				{
					if ( n[i] != this.parentNode )
					{
						$('select', n[i]).val( iVal );
					}
				}
				
				/* Redraw the table */
				oSettings._iDisplayLength = parseInt(iVal, 10);
				_fnCalculateEnd( oSettings );
				
				/* If we have space to show extra rows (backing up from the end point - then do so */
				if ( oSettings._iDisplayEnd == oSettings.aiDisplay.length )
				{
					oSettings._iDisplayStart = oSettings._iDisplayEnd - oSettings._iDisplayLength;
					if ( oSettings._iDisplayStart < 0 )
					{
						oSettings._iDisplayStart = 0;
					}
				}
				
				if ( oSettings._iDisplayLength == -1 )
				{
					oSettings._iDisplayStart = 0;
				}
				
				_fnDraw( oSettings );
			} );
			
			return nLength;
		}
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Feature: Processing incidator
		 */
		
		/*
		 * Function: _fnFeatureHtmlProcessing
		 * Purpose:  Generate the node required for the processing node
		 * Returns:  node
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnFeatureHtmlProcessing ( oSettings )
		{
			var nProcessing = document.createElement( 'div' );
			
			if ( oSettings.sTableId !== '' && typeof oSettings.aanFeatures.r == "undefined" )
			{
				nProcessing.setAttribute( 'id', oSettings.sTableId+'_processing' );
			}
			nProcessing.innerHTML = oSettings.oLanguage.sProcessing;
			nProcessing.className = oSettings.oClasses.sProcessing;
			oSettings.nTable.parentNode.insertBefore( nProcessing, oSettings.nTable );
			
			return nProcessing;
		}
		
		/*
		 * Function: _fnProcessingDisplay
		 * Purpose:  Display or hide the processing indicator
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 *           bool:
		 *   true - show the processing indicator
		 *   false - don't show
		 */
		function _fnProcessingDisplay ( oSettings, bShow )
		{
			if ( oSettings.oFeatures.bProcessing )
			{
				var an = oSettings.aanFeatures.r;
				for ( var i=0, iLen=an.length ; i<iLen ; i++ )
				{
					an[i].style.visibility = bShow ? "visible" : "hidden";
				}
			}
		}
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Support functions
		 */
		
		/*
		 * Function: _fnVisibleToColumnIndex
		 * Purpose:  Covert the index of a visible column to the index in the data array (take account
		 *   of hidden columns)
		 * Returns:  int:i - the data index
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnVisibleToColumnIndex( oSettings, iMatch )
		{
			var iColumn = -1;
			
			for ( var i=0 ; i<oSettings.aoColumns.length ; i++ )
			{
				if ( oSettings.aoColumns[i].bVisible === true )
				{
					iColumn++;
				}
				
				if ( iColumn == iMatch )
				{
					return i;
				}
			}
			
			return null;
		}
		
		/*
		 * Function: _fnColumnIndexToVisible
		 * Purpose:  Covert the index of an index in the data array and convert it to the visible
		 *   column index (take account of hidden columns)
		 * Returns:  int:i - the data index
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnColumnIndexToVisible( oSettings, iMatch )
		{
			var iVisible = -1;
			for ( var i=0 ; i<oSettings.aoColumns.length ; i++ )
			{
				if ( oSettings.aoColumns[i].bVisible === true )
				{
					iVisible++;
				}
				
				if ( i == iMatch )
				{
					return oSettings.aoColumns[i].bVisible === true ? iVisible : null;
				}
			}
			
			return null;
		}
		
		
		/*
		 * Function: _fnNodeToDataIndex
		 * Purpose:  Take a TR element and convert it to an index in aoData
		 * Returns:  int:i - index if found, null if not
		 * Inputs:   object:s - dataTables settings object
		 *           node:n - the TR element to find
		 */
		function _fnNodeToDataIndex( s, n )
		{
			for ( var i=0, iLen=s.aoData.length ; i<iLen ; i++ )
			{
				if ( s.aoData[i] !== null && s.aoData[i].nTr == n )
				{
					return i;
				}
			}
			return null;
		}
		
		/*
		 * Function: _fnVisbleColumns
		 * Purpose:  Get the number of visible columns
		 * Returns:  int:i - the number of visible columns
		 * Inputs:   object:oS - dataTables settings object
		 */
		function _fnVisbleColumns( oS )
		{
			var iVis = 0;
			for ( var i=0 ; i<oS.aoColumns.length ; i++ )
			{
				if ( oS.aoColumns[i].bVisible === true )
				{
					iVis++;
				}
			}
			return iVis;
		}
		
		/*
		 * Function: _fnCalculateEnd
		 * Purpose:  Rcalculate the end point based on the start point
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnCalculateEnd( oSettings )
		{
			if ( oSettings.oFeatures.bPaginate === false )
			{
				oSettings._iDisplayEnd = oSettings.aiDisplay.length;
			}
			else
			{
				/* Set the end point of the display - based on how many elements there are
				 * still to display
				 */
				if ( oSettings._iDisplayStart + oSettings._iDisplayLength > oSettings.aiDisplay.length ||
					   oSettings._iDisplayLength == -1 )
				{
					oSettings._iDisplayEnd = oSettings.aiDisplay.length;
				}
				else
				{
					oSettings._iDisplayEnd = oSettings._iDisplayStart + oSettings._iDisplayLength;
				}
			}
		}
		
		/*
		 * Function: _fnConvertToWidth
		 * Purpose:  Convert a CSS unit width to pixels (e.g. 2em)
		 * Returns:  int:iWidth - width in pixels
		 * Inputs:   string:sWidth - width to be converted
		 *           node:nParent - parent to get the with for (required for
		 *             relative widths) - optional
		 */
		function _fnConvertToWidth ( sWidth, nParent )
		{
			if ( !sWidth || sWidth === null || sWidth === '' )
			{
				return 0;
			}
			
			if ( typeof nParent == "undefined" )
			{
				nParent = document.getElementsByTagName('body')[0];
			}
			
			var iWidth;
			var nTmp = document.createElement( "div" );
			nTmp.style.width = sWidth;
			
			nParent.appendChild( nTmp );
			iWidth = nTmp.offsetWidth;
			nParent.removeChild( nTmp );
			
			return ( iWidth );
		}
		
		/*
		 * Function: _fnCalculateColumnWidths
		 * Purpose:  Calculate the width of columns for the table
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnCalculateColumnWidths ( oSettings )
		{
			var iTableWidth = oSettings.nTable.offsetWidth;
			var iTotalUserIpSize = 0;
			var iTmpWidth;
			var iVisibleColumns = 0;
			var iColums = oSettings.aoColumns.length;
			var i;
			var oHeaders = $('thead:eq(0)>th', oSettings.nTable);
			
			/* Convert any user input sizes into pixel sizes */
			for ( i=0 ; i<iColums ; i++ )
			{
				if ( oSettings.aoColumns[i].bVisible )
				{
					iVisibleColumns++;
					
					if ( oSettings.aoColumns[i].sWidth !== null )
					{
						iTmpWidth = _fnConvertToWidth( oSettings.aoColumns[i].sWidth, 
							oSettings.nTable.parentNode );
						
						/* Total up the user defined widths for later calculations */
						iTotalUserIpSize += iTmpWidth;
						
						oSettings.aoColumns[i].sWidth = iTmpWidth+"px";
					}
				}
			}
			
			/* If the number of columns in the DOM equals the number that we
			 * have to process in dataTables, then we can use the offsets that are
			 * created by the web-browser. No custom sizes can be set in order for
			 * this to happen
			 */
			if ( iColums == oHeaders.length && iTotalUserIpSize === 0 && iVisibleColumns == iColums )
			{
				for ( i=0 ; i<oSettings.aoColumns.length ; i++ )
				{
					oSettings.aoColumns[i].sWidth = oHeaders[i].offsetWidth+"px";
				}
			}
			else
			{
				/* Otherwise we are going to have to do some calculations to get
				 * the width of each column. Construct a 1 row table with the maximum
				 * string sizes in the data, and any user defined widths
				 */
				var nCalcTmp = oSettings.nTable.cloneNode( false );
				nCalcTmp.setAttribute( "id", '' );
				
				var sTableTmp = '<table class="'+nCalcTmp.className+'">';
				var sCalcHead = "<tr>";
				var sCalcHtml = "<tr>";
				
				/* Construct a tempory table which we will inject (invisibly) into
				 * the dom - to let the browser do all the hard word
				 */
				for ( i=0 ; i<iColums ; i++ )
				{
					if ( oSettings.aoColumns[i].bVisible )
					{
						sCalcHead += '<th>'+oSettings.aoColumns[i].sTitle+'</th>';
						
						if ( oSettings.aoColumns[i].sWidth !== null )
						{
							var sWidth = '';
							if ( oSettings.aoColumns[i].sWidth !== null )
							{
								sWidth = ' style="width:'+oSettings.aoColumns[i].sWidth+';"';
							}
							
							sCalcHtml += '<td'+sWidth+' tag_index="'+i+'">'+fnGetMaxLenString( oSettings, i)+'</td>';
						}
						else
						{
							sCalcHtml += '<td tag_index="'+i+'">'+fnGetMaxLenString( oSettings, i)+'</td>';
						}
					}
				}
				
				sCalcHead += "</tr>";
				sCalcHtml += "</tr>";
				
				/* Create the tmp table node (thank you jQuery) */
				nCalcTmp = $( sTableTmp + sCalcHead + sCalcHtml +'</table>' )[0];
				nCalcTmp.style.width = iTableWidth + "px";
				nCalcTmp.style.visibility = "hidden";
				nCalcTmp.style.position = "absolute"; /* Try to aviod scroll bar */
				
				oSettings.nTable.parentNode.appendChild( nCalcTmp );
				
				var oNodes = $("tr:eq(1)>td", nCalcTmp);
				var iIndex;
				
				/* Gather in the browser calculated widths for the rows */
				for ( i=0 ; i<oNodes.length ; i++ )
				{
					iIndex = oNodes[i].getAttribute('tag_index');
					
					var iContentWidth = $("td", nCalcTmp).eq(i).width();
					var iSetWidth = oSettings.aoColumns[i].sWidth ? 
						oSettings.aoColumns[i].sWidth.slice(0, -2) : 0;
					oSettings.aoColumns[iIndex].sWidth = Math.max(iContentWidth, iSetWidth) + "px";
				}
				
				oSettings.nTable.parentNode.removeChild( nCalcTmp );
			}
		}
		
		/*
		 * Function: fnGetMaxLenString
		 * Purpose:  Get the maximum strlen for each data column
		 * Returns:  string: - max strlens for each column
		 * Inputs:   object:oSettings - dataTables settings object
		 *           int:iCol - column of interest
		 */
		function fnGetMaxLenString( oSettings, iCol )
		{
			var iMax = 0;
			var iMaxIndex = -1;
			
			for ( var i=0 ; i<oSettings.aoData.length ; i++ )
			{
				if ( oSettings.aoData[i]._aData[iCol].length > iMax )
				{
					iMax = oSettings.aoData[i]._aData[iCol].length;
					iMaxIndex = i;
				}
			}
			
			if ( iMaxIndex >= 0 )
			{
				return oSettings.aoData[iMaxIndex]._aData[iCol];
			}
			return '';
		}
		
		/*
		 * Function: _fnArrayCmp
		 * Purpose:  Compare two arrays
		 * Returns:  0 if match, 1 if length is different, 2 if no match
		 * Inputs:   array:aArray1 - first array
		 *           array:aArray2 - second array
		 */
		function _fnArrayCmp( aArray1, aArray2 )
		{
			if ( aArray1.length != aArray2.length )
			{
				return 1;
			}
			
			for ( var i=0 ; i<aArray1.length ; i++ )
			{
				if ( aArray1[i] != aArray2[i] )
				{
					return 2;
				}
			}
			
			return 0;
		}
		
		/*
		 * Function: _fnDetectType
		 * Purpose:  Get the sort type based on an input string
		 * Returns:  string: - type (defaults to 'string' if no type can be detected)
		 * Inputs:   string:sData - data we wish to know the type of
		 * Notes:    This function makes use of the DataTables plugin objct _oExt 
		 *   (.aTypes) such that new types can easily be added.
		 */
		function _fnDetectType( sData )
		{
			var aTypes = _oExt.aTypes;
			var iLen = aTypes.length;
			
			for ( var i=0 ; i<iLen ; i++ )
			{
				var sType = aTypes[i]( sData );
				if ( sType !== null )
				{
					return sType;
				}
			}
			
			return 'string';
		}
		
		/*
		 * Function: _fnSettingsFromNode
		 * Purpose:  Return the settings object for a particular table
		 * Returns:  object: Settings object - or null if not found
		 * Inputs:   node:nTable - table we are using as a dataTable
		 */
		function _fnSettingsFromNode ( nTable )
		{
			for ( var i=0 ; i<_aoSettings.length ; i++ )
			{
				if ( _aoSettings[i].nTable == nTable )
				{
					return _aoSettings[i];
				}
			}
			
			return null;
		}
		
		/*
		 * Function: _fnGetDataMaster
		 * Purpose:  Return an array with the full table data
		 * Returns:  array array:aData - Master data array
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnGetDataMaster ( oSettings )
		{
			var aData = [];
			var iLen = oSettings.aoData.length;
			for ( var i=0 ; i<iLen; i++ )
			{
				if ( oSettings.aoData[i] === null )
				{
					aData.push( null );
				}
				else
				{
					aData.push( oSettings.aoData[i]._aData );
				}
			}
			return aData;
		}
		
		/*
		 * Function: _fnGetTrNodes
		 * Purpose:  Return an array with the TR nodes for the table
		 * Returns:  array: - TR array
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnGetTrNodes ( oSettings )
		{
			var aNodes = [];
			var iLen = oSettings.aoData.length;
			for ( var i=0 ; i<iLen ; i++ )
			{
				if ( oSettings.aoData[i] === null )
				{
					aNodes.push( null );
				}
				else
				{
					aNodes.push( oSettings.aoData[i].nTr );
				}
			}
			return aNodes;
		}
		
		/*
		 * Function: _fnGetTdNodes
		 * Purpose:  Return an array with the TD nodes for the table
		 * Returns:  array: - TD array
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnGetTdNodes ( oSettings )
		{
			var nTrs = _fnGetTrNodes( oSettings );
			var nTds = [], nTd;
			var anReturn = [];
			var iCorrector;
			var iRow, iRows, iColumn, iColumns;
			
			for ( iRow=0, iRows=nTrs.length ; iRow<iRows ; iRow++ )
			{
				nTds = [];
				for ( iColumn=0, iColumns=nTrs[iRow].childNodes.length ; iColumn<iColumns ; iColumn++ )
				{
					nTd = nTrs[iRow].childNodes[iColumn];
					if ( nTd.nodeName == "TD" )
					{
						nTds.push( nTd );
					}
				}
				
				iCorrector = 0;
				for ( iColumn=0, iColumns=oSettings.aoColumns.length ; iColumn<iColumns ; iColumn++ )
				{
					if ( oSettings.aoColumns[iColumn].bVisible )
					{
						anReturn.push( nTds[iColumn-iCorrector] );
					}
					else
					{
						anReturn.push( oSettings.aoData[iRow]._anHidden[iColumn] );
						iCorrector++;
					}
				}
			}
			return anReturn;
		}
		
		/*
		 * Function: _fnEscapeRegex
		 * Purpose:  scape a string stuch that it can be used in a regular expression
		 * Returns:  string: - escaped string
		 * Inputs:   string:sVal - string to escape
		 */
		function _fnEscapeRegex ( sVal )
		{
			var acEscape = [ '/', '.', '*', '+', '?', '|', '(', ')', '[', ']', '{', '}', '\\', '$', '^' ];
		  var reReplace = new RegExp( '(\\' + acEscape.join('|\\') + ')', 'g' );
		  return sVal.replace(reReplace, '\\$1');
		}
		
		/*
		 * Function: _fnReOrderIndex
		 * Purpose:  Figure out how to reorder a display list
		 * Returns:  array int:aiReturn - index list for reordering
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnReOrderIndex ( oSettings, sColumns )
		{
			var aColumns = sColumns.split(',');
			var aiReturn = [];
			
			for ( var i=0, iLen=oSettings.aoColumns.length ; i<iLen ; i++ )
			{
				for ( var j=0 ; j<iLen ; j++ )
				{
					if ( oSettings.aoColumns[i].sName == aColumns[j] )
					{
						aiReturn.push( j );
						break;
					}
				}
			}
			
			return aiReturn;
		}
		
		/*
		 * Function: _fnColumnOrdering
		 * Purpose:  Get the column ordering that DataTables expects
		 * Returns:  string: - comma separated list of names
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnColumnOrdering ( oSettings )
		{
			var sNames = '';
			for ( var i=0, iLen=oSettings.aoColumns.length ; i<iLen ; i++ )
			{
				sNames += oSettings.aoColumns[i].sName+',';
			}
			if ( sNames.length == iLen )
			{
				return "";
			}
			return sNames.slice(0, -1);
		}
		
		/*
		 * Function: _fnClearTable
		 * Purpose:  Nuke the table
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnClearTable( oSettings )
		{
			oSettings.aoData.length = 0;
			oSettings.aiDisplayMaster.length = 0;
			oSettings.aiDisplay.length = 0;
			_fnCalculateEnd( oSettings );
		}
		
		/*
		 * Function: _fnSaveState
		 * Purpose:  Save the state of a table in a cookie such that the page can be reloaded
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 */
		function _fnSaveState ( oSettings )
		{
			if ( !oSettings.oFeatures.bStateSave )
			{
				return;
			}
			
			/* Store the interesting variables */
			var i;
			var sValue = "{";
			sValue += '"iStart": '+oSettings._iDisplayStart+',';
			sValue += '"iEnd": '+oSettings._iDisplayEnd+',';
			sValue += '"iLength": '+oSettings._iDisplayLength+',';
			sValue += '"sFilter": "'+oSettings.oPreviousSearch.sSearch.replace('"','\\"')+'",';
			sValue += '"sFilterEsc": '+oSettings.oPreviousSearch.bEscapeRegex+',';
			
			sValue += '"aaSorting": [ ';
			for ( i=0 ; i<oSettings.aaSorting.length ; i++ )
			{
				sValue += "["+oSettings.aaSorting[i][0]+",'"+oSettings.aaSorting[i][1]+"'],";
			}
			sValue = sValue.substring(0, sValue.length-1);
			sValue += "],";
			
			sValue += '"aaSearchCols": [ ';
			for ( i=0 ; i<oSettings.aoPreSearchCols.length ; i++ )
			{
				sValue += "['"+oSettings.aoPreSearchCols[i].sSearch.replace("'","\'")+
					"',"+oSettings.aoPreSearchCols[i].bEscapeRegex+"],";
			}
			sValue = sValue.substring(0, sValue.length-1);
			sValue += "],";
			
			sValue += '"abVisCols": [ ';
			for ( i=0 ; i<oSettings.aoColumns.length ; i++ )
			{
				sValue += oSettings.aoColumns[i].bVisible+",";
			}
			sValue = sValue.substring(0, sValue.length-1);
			sValue += "]";
			
			sValue += "}";
			_fnCreateCookie( "SpryMedia_DataTables_"+oSettings.sInstance, sValue, 
				oSettings.iCookieDuration );
		}
		
		/*
		 * Function: _fnLoadState
		 * Purpose:  Attempt to load a saved table state from a cookie
		 * Returns:  -
		 * Inputs:   object:oSettings - dataTables settings object
		 *           object:oInit - DataTables init object so we can override settings
		 */
		function _fnLoadState ( oSettings, oInit )
		{
			if ( !oSettings.oFeatures.bStateSave )
			{
				return;
			}
			
			var oData;
			var sData = _fnReadCookie( "SpryMedia_DataTables_"+oSettings.sInstance );
			if ( sData !== null && sData !== '' )
			{
				/* Try/catch the JSON eval - if it is bad then we ignore it */
				try
				{
					/* Use the JSON library for safety - if it is available */
					if ( typeof JSON == 'object' && typeof JSON.parse == 'function' )
					{
						/* DT 1.4.0 used single quotes for a string - JSON.parse doesn't allow this and throws
						 * an error. So for now we can do this. This can be removed in future it is just to
						 * allow the tranfrer to 1.4.1+ to occur
						 */
						oData = JSON.parse( sData.replace(/'/g, '"') );
					}
					else
					{
						oData = eval( '('+sData+')' );
					}
				}
				catch( e )
				{
					return;
				}
				
				/* Restore key features */
				oSettings._iDisplayStart = oData.iStart;
				oSettings.iInitDisplayStart = oData.iStart;
				oSettings._iDisplayEnd = oData.iEnd;
				oSettings._iDisplayLength = oData.iLength;
				oSettings.oPreviousSearch.sSearch = oData.sFilter;
				oSettings.aaSorting = oData.aaSorting.slice();
				oSettings.saved_aaSorting = oData.aaSorting.slice();
				
				/* Search filtering - global reference added in 1.4.1 */
				if ( typeof oData.sFilterEsc != 'undefined' )
				{
					oSettings.oPreviousSearch.bEscapeRegex = oData.sFilterEsc;
				}
				
				/* Column filtering - added in 1.5.0 beta 6 */
				if ( typeof oData.aaSearchCols != 'undefined' )
				{
					for ( var i=0 ; i<oData.aaSearchCols.length ; i++ )
					{
						oSettings.aoPreSearchCols[i] = {
							"sSearch": oData.aaSearchCols[i][0],
							"bEscapeRegex": oData.aaSearchCols[i][1]
						};
					}
				}
				
				/* Column visibility state - added in 1.5.0 beta 10 */
				if ( typeof oData.abVisCols != 'undefined' )
				{
					/* Pass back visibiliy settings to the init handler, but to do not here override
					 * the init object that the user might have passed in
					 */
					oInit.saved_aoColumns = [];
					for ( i=0 ; i<oData.abVisCols.length ; i++ )
					{
						oInit.saved_aoColumns[i] = {};
						oInit.saved_aoColumns[i].bVisible = oData.abVisCols[i];
					}
				}
			}
		}
		
		/*
		 * Function: _fnCreateCookie
		 * Purpose:  Create a new cookie with a value to store the state of a table
		 * Returns:  -
		 * Inputs:   string:sName - name of the cookie to create
		 *           string:sValue - the value the cookie should take
		 *           int:iSecs - duration of the cookie
		 */
		function _fnCreateCookie ( sName, sValue, iSecs )
		{
			var date = new Date();
			date.setTime( date.getTime()+(iSecs*1000) );
			
			/* 
			 * Shocking but true - it would appear IE has major issues with having the path being
			 * set to anything but root. We need the cookie to be available based on the path, so we
			 * have to append the pathname to the cookie name. Appalling.
			 */
			sName += '_'+window.location.pathname.replace(/[\/:]/g,"").toLowerCase();
			
			document.cookie = sName+"="+encodeURIComponent(sValue)+
				"; expires="+date.toGMTString()+"; path=/";
		}
		
		/*
		 * Function: _fnReadCookie
		 * Purpose:  Read an old cookie to get a cookie with an old table state
		 * Returns:  string: - contents of the cookie - or null if no cookie with that name found
		 * Inputs:   string:sName - name of the cookie to read
		 */
		function _fnReadCookie ( sName )
		{
			var sNameEQ = sName +'_'+ window.location.pathname.replace(/[\/:]/g,"").toLowerCase() + "=";
			var sCookieContents = document.cookie.split(';');
			
			for( var i=0 ; i<sCookieContents.length ; i++ )
			{
				var c = sCookieContents[i];
				
				while (c.charAt(0)==' ')
				{
					c = c.substring(1,c.length);
				}
				
				if (c.indexOf(sNameEQ) === 0)
				{
					return decodeURIComponent( c.substring(sNameEQ.length,c.length) );
				}
			}
			return null;
		}
		
		/*
		 * Function: _fnGetUniqueThs
		 * Purpose:  Get an array of unique th elements, one for each column
		 * Returns:  array node:aReturn - list of unique ths
		 * Inputs:   node:nThead - The thead element for the table
		 */
		function _fnGetUniqueThs ( nThead )
		{
			var nTrs = nThead.getElementsByTagName('tr');
			
			/* Nice simple case */
			if ( nTrs.length == 1 )
			{
				return nTrs[0].getElementsByTagName('th');
			}
			
			/* Otherwise we need to figure out the layout array to get the nodes */
			var aLayout = [], aReturn = [];
			var ROWSPAN = 2, COLSPAN = 3, TDELEM = 4;
			var i, j, k, iLen, jLen, iColumnShifted;
			var fnShiftCol = function ( a, i, j ) {
				while ( typeof a[i][j] != 'undefined' ) {
					j++;
				}
				return j;
			};
			var fnAddRow = function ( i ) {
				if ( typeof aLayout[i] == 'undefined' ) {
					aLayout[i] = [];
				}
			};
			
			/* Calculate a layout array */
			for ( i=0, iLen=nTrs.length ; i<iLen ; i++ )
			{
				fnAddRow( i );
				var iColumn = 0;
				var nTds = [];
				
				for ( j=0, jLen=nTrs[i].childNodes.length ; j<jLen ; j++ )
				{
					if ( nTrs[i].childNodes[j].nodeName == "TD" || nTrs[i].childNodes[j].nodeName == "TH" )
					{
						nTds.push( nTrs[i].childNodes[j] );
					}
				}
				
				for ( j=0, jLen=nTds.length ; j<jLen ; j++ )
				{
					var iColspan = nTds[j].getAttribute('colspan') * 1;
					var iRowspan = nTds[j].getAttribute('rowspan') * 1;
					
					if ( !iColspan || iColspan===0 || iColspan===1 )
					{
						iColumnShifted = fnShiftCol( aLayout, i, iColumn );
						aLayout[i][iColumnShifted] = (nTds[j].nodeName=="TD") ? TDELEM : nTds[j];
						if ( iRowspan || iRowspan===0 || iRowspan===1 )
						{
							for ( k=1 ; k<iRowspan ; k++ )
							{
								fnAddRow( i+k );
								aLayout[i+k][iColumnShifted] = ROWSPAN;
							}
						}
						iColumn++;
					}
					else
					{
						iColumnShifted = fnShiftCol( aLayout, i, iColumn );
						for ( k=0 ; k<iColspan ; k++ )
						{
							aLayout[i][iColumnShifted+k] = COLSPAN;
						}
						iColumn += iColspan;
					}
				}
			}
			
			/* Convert the layout array into a node array
			 * Note the use of aLayout[0] in the outloop, we want the outer loop to occur the same
			 * number of times as there are columns. Unusual having nested loops this way around
			 * but is what we need here.
			 */
			for ( i=0, iLen=aLayout[0].length ; i<iLen ; i++ )
			{
				for ( j=0, jLen=aLayout.length ; j<jLen ; j++ )
				{
					if ( typeof aLayout[j][i] == 'object' )
					{
						aReturn.push( aLayout[j][i] );
					}
				}
			}
			
			return aReturn;
		}
		
		/*
		 * Function: _fnMap
		 * Purpose:  See if a property is defined on one object, if so assign it to the other object
		 * Returns:  - (done by reference)
		 * Inputs:   object:oRet - target object
		 *           object:oSrc - source object
		 *           string:sName - property
		 *           string:sMappedName - name to map too - optional, sName used if not given
		 */
		function _fnMap( oRet, oSrc, sName, sMappedName )
		{
			if ( typeof sMappedName == 'undefined' )
			{
				sMappedName = sName;
			}
			if ( typeof oSrc[sName] != 'undefined' )
			{
				oRet[sMappedName] = oSrc[sName];
			}
		}
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - API
		 * 
		 * I'm not overly happy with this solution - I'd much rather that there was a way of getting
		 * a list of all the private functions and do what we need to dynamically - but that doesn't
		 * appear to be possible. Bonkers. A better solution would be to provide a 'bind' type object
		 * To do - bind type method in DTs 2.x.
		 */
		this.oApi._fnInitalise = _fnInitalise;
		this.oApi._fnLanguageProcess = _fnLanguageProcess;
		this.oApi._fnAddColumn = _fnAddColumn;
		this.oApi._fnAddData = _fnAddData;
		this.oApi._fnGatherData = _fnGatherData;
		this.oApi._fnDrawHead = _fnDrawHead;
		this.oApi._fnDraw = _fnDraw;
		this.oApi._fnAjaxUpdate = _fnAjaxUpdate;
		this.oApi._fnAddOptionsHtml = _fnAddOptionsHtml;
		this.oApi._fnFeatureHtmlFilter = _fnFeatureHtmlFilter;
		this.oApi._fnFeatureHtmlInfo = _fnFeatureHtmlInfo;
		this.oApi._fnFeatureHtmlPaginate = _fnFeatureHtmlPaginate;
		this.oApi._fnPageChange = _fnPageChange;
		this.oApi._fnFeatureHtmlLength = _fnFeatureHtmlLength;
		this.oApi._fnFeatureHtmlProcessing = _fnFeatureHtmlProcessing;
		this.oApi._fnProcessingDisplay = _fnProcessingDisplay;
		this.oApi._fnFilterComplete = _fnFilterComplete;
		this.oApi._fnFilterColumn = _fnFilterColumn;
		this.oApi._fnFilter = _fnFilter;
		this.oApi._fnSortingClasses = _fnSortingClasses;
		this.oApi._fnVisibleToColumnIndex = _fnVisibleToColumnIndex;
		this.oApi._fnColumnIndexToVisible = _fnColumnIndexToVisible;
		this.oApi._fnNodeToDataIndex = _fnNodeToDataIndex;
		this.oApi._fnVisbleColumns = _fnVisbleColumns;
		this.oApi._fnBuildSearchArray = _fnBuildSearchArray;
		this.oApi._fnDataToSearch = _fnDataToSearch;
		this.oApi._fnCalculateEnd = _fnCalculateEnd;
		this.oApi._fnConvertToWidth = _fnConvertToWidth;
		this.oApi._fnCalculateColumnWidths = _fnCalculateColumnWidths;
		this.oApi._fnArrayCmp = _fnArrayCmp;
		this.oApi._fnDetectType = _fnDetectType;
		this.oApi._fnGetDataMaster = _fnGetDataMaster;
		this.oApi._fnGetTrNodes = _fnGetTrNodes;
		this.oApi._fnGetTdNodes = _fnGetTdNodes;
		this.oApi._fnEscapeRegex = _fnEscapeRegex;
		this.oApi._fnReOrderIndex = _fnReOrderIndex;
		this.oApi._fnColumnOrdering = _fnColumnOrdering;
		this.oApi._fnClearTable = _fnClearTable;
		this.oApi._fnSaveState = _fnSaveState;
		this.oApi._fnLoadState = _fnLoadState;
		this.oApi._fnCreateCookie = _fnCreateCookie;
		this.oApi._fnReadCookie = _fnReadCookie;
		this.oApi._fnGetUniqueThs = _fnGetUniqueThs;
		this.oApi._fnReDraw = _fnReDraw;
		
		/* Want to be able to reference "this" inside the this.each function */
		var _that = this;
		
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * Section - Constructor
		 */
		return this.each(function()
		{
			var i=0, iLen, j, jLen;
			
			/* Sanity check that we are not re-initialising a table - if we are, alert an error */
			for ( i=0, iLen=_aoSettings.length ; i<iLen ; i++ )
			{
				if ( _aoSettings[i].nTable == this )
				{
					alert( "DataTables warning: Unable to re-initialise DataTable. "+
						"Please use the API to make any configuration changes required." );
					return _aoSettings[i];
				}
			}
			
			/* Make a complete and independent copy of the settings object */
			var oSettings = new classSettings();
			_aoSettings.push( oSettings );
			
			var bInitHandedOff = false;
			var bUsePassedData = false;
			
			/* Set the id */
			var sId = this.getAttribute( 'id' );
			if ( sId !== null )
			{
				oSettings.sTableId = sId;
				oSettings.sInstance = sId;
			}
			else
			{
				oSettings.sInstance = _oExt._oExternConfig.iNextUnique ++;
			}
			
			/* Set the table node */
			oSettings.nTable = this;
			
			/* Bind the API functions to the settings, so we can perform actions whenever oSettings is
			 * available
			 */
			oSettings.oApi = _that.oApi;
			
			/* Store the features that we have available */
			if ( typeof oInit != 'undefined' && oInit !== null )
			{
				_fnMap( oSettings.oFeatures, oInit, "bPaginate" );
				_fnMap( oSettings.oFeatures, oInit, "bLengthChange" );
				_fnMap( oSettings.oFeatures, oInit, "bFilter" );
				_fnMap( oSettings.oFeatures, oInit, "bSort" );
				_fnMap( oSettings.oFeatures, oInit, "bInfo" );
				_fnMap( oSettings.oFeatures, oInit, "bProcessing" );
				_fnMap( oSettings.oFeatures, oInit, "bAutoWidth" );
				_fnMap( oSettings.oFeatures, oInit, "bSortClasses" );
				_fnMap( oSettings.oFeatures, oInit, "bServerSide" );
				_fnMap( oSettings, oInit, "asStripClasses" );
				_fnMap( oSettings, oInit, "fnRowCallback" );
				_fnMap( oSettings, oInit, "fnHeaderCallback" );
				_fnMap( oSettings, oInit, "fnFooterCallback" );
				_fnMap( oSettings, oInit, "fnInitComplete" );
				_fnMap( oSettings, oInit, "fnServerData" );
				_fnMap( oSettings, oInit, "aaSorting" );
				_fnMap( oSettings, oInit, "aaSortingFixed" );
				_fnMap( oSettings, oInit, "sPaginationType" );
				_fnMap( oSettings, oInit, "sAjaxSource" );
				_fnMap( oSettings, oInit, "iCookieDuration" );
				_fnMap( oSettings, oInit, "sDom" );
				_fnMap( oSettings, oInit, "oSearch", "oPreviousSearch" );
				_fnMap( oSettings, oInit, "aoSearchCols", "aoPreSearchCols" );
				_fnMap( oSettings, oInit, "iDisplayLength", "_iDisplayLength" );
				_fnMap( oSettings, oInit, "bJQueryUI", "bJUI" );
				
				if ( typeof oInit.fnDrawCallback == 'function' )
				{
					/* Add user given callback function to array */
					oSettings.aoDrawCallback.push( {
						"fn": oInit.fnDrawCallback,
						"sName": "user"
					} );
				}
				
				if ( oSettings.oFeatures.bServerSide && oSettings.oFeatures.bSort &&
					   oSettings.oFeatures.bSortClasses )
				{
					/* Enable sort classes for server-side processing. Safe to do it here, since server-side
					 * processing must be enabled by the developer
					 */
					oSettings.aoDrawCallback.push( {
						"fn": _fnSortingClasses,
						"sName": "server_side_sort_classes"
					} );
				}
				
				if ( typeof oInit.bJQueryUI != 'undefined' && oInit.bJQueryUI )
				{
					/* Use the JUI classes object for display. You could clone the oStdClasses object if 
					 * you want to have multiple tables with multiple independent classes 
					 */
					oSettings.oClasses = _oExt.oJUIClasses;
					
					if ( typeof oInit.sDom == 'undefined' )
					{
						/* Set the DOM to use a layout suitable for jQuery UI's theming */
						oSettings.sDom = '<"H"lfr>t<"F"ip>';
					}
				}
				
				if ( typeof oInit.iDisplayStart != 'undefined' && 
				     typeof oSettings.iInitDisplayStart == 'undefined' )
				{
					/* Display start point, taking into account the save saving */
					oSettings.iInitDisplayStart = oInit.iDisplayStart;
					oSettings._iDisplayStart = oInit.iDisplayStart;
				}
				
				/* Must be done after everything which can be overridden by a cookie! */
				if ( typeof oInit.bStateSave != 'undefined' )
				{
					oSettings.oFeatures.bStateSave = oInit.bStateSave;
					_fnLoadState( oSettings, oInit );
					oSettings.aoDrawCallback.push( {
						"fn": _fnSaveState,
						"sName": "state_save"
					} );
				}
				
				if ( typeof oInit.aaData != 'undefined' )
				{
					bUsePassedData = true;
				}
				
				/* Backwards compatability */
				/* aoColumns / aoData - remove at some point... */
				if ( typeof oInit != 'undefined' && typeof oInit.aoData != 'undefined' )
				{
					oInit.aoColumns = oInit.aoData;
				}
				
				/* Language definitions */
				if ( typeof oInit.oLanguage != 'undefined' )
				{
					if ( typeof oInit.oLanguage.sUrl != 'undefined' && oInit.oLanguage.sUrl !== "" )
					{
						/* Get the language definitions from a file */
						oSettings.oLanguage.sUrl = oInit.oLanguage.sUrl;
						$.getJSON( oSettings.oLanguage.sUrl, null, function( json ) { 
							_fnLanguageProcess( oSettings, json, true ); } );
						bInitHandedOff = true;
					}
					else
					{
						_fnLanguageProcess( oSettings, oInit.oLanguage, false );
					}
				}
				/* Warning: The _fnLanguageProcess function is async to the remainder of this function due
				 * to the XHR. We use _bInitialised in _fnLanguageProcess() to check this the processing 
				 * below is complete. The reason for spliting it like this is optimisation - we can fire
				 * off the XHR (if needed) and then continue processing the data.
				 */
			}
			else
			{
				/* Create a dummy object for quick manipulation later on. */
				oInit = {};
			}
				
			/* Add the strip classes now that we know which classes to apply - unless overruled */
			if ( typeof oInit.asStripClasses == 'undefined' )
			{
				oSettings.asStripClasses.push( oSettings.oClasses.sStripOdd );
				oSettings.asStripClasses.push( oSettings.oClasses.sStripEven );
			}
			
			/* See if we should load columns automatically or use defined ones - a bit messy this... */
			var nThead = this.getElementsByTagName('thead');
			var nThs = nThead.length===0 ? null : _fnGetUniqueThs( nThead[0] );
			var bUseCols = typeof oInit.aoColumns != 'undefined';
			
			for ( i=0, iLen=bUseCols ? oInit.aoColumns.length : nThs.length ; i<iLen ; i++ )
			{
				var oCol = bUseCols ? oInit.aoColumns[i] : null;
				var nTh = nThs ? nThs[i] : null;
				
				/* Check if we have column visibilty state to restore, and also that the length of the 
				 * state saved columns matches the currently know number of columns
				 */
				if ( typeof oInit.saved_aoColumns != 'undefined' && oInit.saved_aoColumns.length == iLen )
				{
					if ( oCol === null )
					{
						oCol = {};
					}
					oCol.bVisible = oInit.saved_aoColumns[i].bVisible;
				}
				
				_fnAddColumn( oSettings, oCol, nTh );
			}
			
			/* Check the aaSorting array */
			for ( i=0, iLen=oSettings.aaSorting.length ; i<iLen ; i++ )
			{
				var oColumn = oSettings.aoColumns[ oSettings.aaSorting[i][0] ];
				
				/* Add a default sorting index */
				if ( typeof oSettings.aaSorting[i][2] == 'undefined' )
				{
					oSettings.aaSorting[i][2] = 0;
				}
				
				/* If aaSorting is not defined, then we use the first indicator in asSorting */
				if ( typeof oInit.aaSorting == "undefined" && 
						 typeof oSettings.saved_aaSorting == "undefined" )
				{
					oSettings.aaSorting[i][1] = oColumn.asSorting[0];
				}
				
				/* Set the current sorting index based on aoColumns.asSorting */
				for ( j=0, jLen=oColumn.asSorting.length ; j<jLen ; j++ )
				{
					if ( oSettings.aaSorting[i][1] == oColumn.asSorting[j] )
					{
						oSettings.aaSorting[i][2] = j;
						break;
					}
				}
			}
			
			/* Sanity check that there is a thead and tfoot. If not let's just create them */
			if ( this.getElementsByTagName('thead').length === 0 )
			{
				this.appendChild( document.createElement( 'thead' ) );
			}
			
			if ( this.getElementsByTagName('tbody').length === 0 )
			{
				this.appendChild( document.createElement( 'tbody' ) );
			}
			
			/* Check if there is data passing into the constructor */
			if ( bUsePassedData )
			{
				for ( i=0 ; i<oInit.aaData.length ; i++ )
				{
					_fnAddData( oSettings, oInit.aaData[ i ] );
				}
			}
			else
			{
				/* Grab the data from the page */
				_fnGatherData( oSettings );
			}
			
			/* Copy the data index array */
			oSettings.aiDisplay = oSettings.aiDisplayMaster.slice();
			
			/* Calculate sizes for columns */
			if ( oSettings.oFeatures.bAutoWidth )
			{
				_fnCalculateColumnWidths( oSettings );
			}
			
			/* Initialisation complete - table can be drawn */
			oSettings.bInitialised = true;
			
			/* Check if we need to initialise the table (it might not have been handed off to the
			 * language processor)
			 */
			if ( bInitHandedOff === false )
			{
				_fnInitalise( oSettings );
			}
		});
	};
})(jQuery);
