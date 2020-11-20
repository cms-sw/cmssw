webpackHotUpdate_N_E("pages/index",{

/***/ "./components/overlayWithAnotherPlot/index.tsx":
/*!*****************************************************!*\
  !*** ./components/overlayWithAnotherPlot/index.tsx ***!
  \*****************************************************/
/*! exports provided: OverlayWithAnotherPlot */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "OverlayWithAnotherPlot", function() { return OverlayWithAnotherPlot; });
/* harmony import */ var _babel_runtime_helpers_esm_toConsumableArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/toConsumableArray */ "./node_modules/@babel/runtime/helpers/esm/toConsumableArray.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd/lib/modal/Modal */ "./node_modules/antd/lib/modal/Modal.js");
/* harmony import */ var antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../containers/display/utils */ "./containers/display/utils.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _containers_display_content_folderPath__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../../containers/display/content/folderPath */ "./containers/display/content/folderPath.tsx");
/* harmony import */ var clean_deep__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! clean-deep */ "./node_modules/clean-deep/src/index.js");
/* harmony import */ var clean_deep__WEBPACK_IMPORTED_MODULE_11___default = /*#__PURE__*/__webpack_require__.n(clean_deep__WEBPACK_IMPORTED_MODULE_11__);
/* harmony import */ var _containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../containers/search/styledComponents */ "./containers/search/styledComponents.tsx");



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/overlayWithAnotherPlot/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2__["createElement"];











var OverlayWithAnotherPlot = function OverlayWithAnotherPlot(_ref) {
  _s();

  var visible = _ref.visible,
      setOpenOverlayWithAnotherPlotModal = _ref.setOpenOverlayWithAnotherPlotModal;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_2__["useState"]({
    folder_path: '',
    name: ''
  }),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState, 2),
      overlaidPlots = _React$useState2[0],
      setOverlaidPlots = _React$useState2[1];

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_2__["useState"]([]),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState3, 2),
      folders = _React$useState4[0],
      setFolders = _React$useState4[1];

  var _React$useState5 = react__WEBPACK_IMPORTED_MODULE_2__["useState"](''),
      _React$useState6 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState5, 2),
      currentFolder = _React$useState6[0],
      setCurrentFolder = _React$useState6[1];

  var _React$useState7 = react__WEBPACK_IMPORTED_MODULE_2__["useState"]({}),
      _React$useState8 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState7, 2),
      plot = _React$useState8[0],
      setPlot = _React$useState8[1];

  var _React$useState9 = react__WEBPACK_IMPORTED_MODULE_2__["useState"](),
      _React$useState10 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState9, 2),
      height = _React$useState10[0],
      setHeight = _React$useState10[1];

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_4__["useRouter"])();
  var query = router.query;

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_2__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__["store"]),
      updated_by_not_older_than = _React$useContext.updated_by_not_older_than;

  var params = {
    dataset_name: query.dataset_name,
    run_number: query.run_number,
    notOlderThan: updated_by_not_older_than,
    folders_path: overlaidPlots.folder_path,
    plot_name: overlaidPlots.name
  };
  var api = Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_6__["choose_api"])(params);
  var data_get_by_mount = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__["useRequest"])(api, {}, [overlaidPlots.folder_path]);
  react__WEBPACK_IMPORTED_MODULE_2__["useEffect"](function () {
    var copy = Object(_babel_runtime_helpers_esm_toConsumableArray__WEBPACK_IMPORTED_MODULE_0__["default"])(folders);

    var index = folders.indexOf(currentFolder);

    if (index >= 0) {
      var rest = copy.splice(0, index + 1);
      setFolders(rest);
      var joinderFolders = rest.join('/');
      setOverlaidPlots({
        folder_path: joinderFolders,
        name: ''
      });
    } else {
      copy.push(currentFolder); //we're cleaning copy array, because we want to delete empty string. 
      // We need to remove it because when we're joining array with empty string 
      // we're getting a string with '/' in the beginning.

      var cleaned_array = clean_deep__WEBPACK_IMPORTED_MODULE_11___default()(copy) ? clean_deep__WEBPACK_IMPORTED_MODULE_11___default()(copy) : [];
      setFolders(cleaned_array);

      var _joinderFolders = copy.join('/');

      if (cleaned_array.length === 0) {
        setOverlaidPlots({
          folder_path: '',
          name: ''
        });
      }

      setOverlaidPlots({
        folder_path: _joinderFolders,
        name: ''
      });
    }
  }, [currentFolder]);
  var modalRef = react__WEBPACK_IMPORTED_MODULE_2__["useRef"](null);
  var data = data_get_by_mount.data;
  var folders_or_plots = data ? data.data : [];

  var changeFolderPathByBreadcrumb = function changeFolderPathByBreadcrumb(item) {
    var folders_from_breadcrumb = item.folder_path.split('/');
    var cleaned_folders_array = clean_deep__WEBPACK_IMPORTED_MODULE_11___default()(folders_from_breadcrumb) ? clean_deep__WEBPACK_IMPORTED_MODULE_11___default()(folders_from_breadcrumb) : [];
    setFolders(cleaned_folders_array);

    if (cleaned_folders_array.length > 0) {
      setCurrentFolder(cleaned_folders_array[cleaned_folders_array.length - 1]);
    } else {
      setCurrentFolder('');
    }
  };

  return __jsx(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3___default.a, {
    visible: visible,
    onCancel: function onCancel() {
      setOpenOverlayWithAnotherPlotModal(false);
      setCurrentFolder('');
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 88,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    gutter: 16,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 95,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
    style: {
      padding: 8
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 96,
      columnNumber: 9
    }
  }, __jsx(_containers_display_content_folderPath__WEBPACK_IMPORTED_MODULE_10__["FolderPath"], {
    folder_path: overlaidPlots.folder_path,
    changeFolderPathByBreadcrumb: changeFolderPathByBreadcrumb,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 97,
      columnNumber: 11
    }
  })), !data_get_by_mount.isLoading && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    style: {
      width: '100%',
      flex: '1 1 auto'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 101,
      columnNumber: 11
    }
  }, folders_or_plots.map(function (folder_or_plot) {
    return __jsx(react__WEBPACK_IMPORTED_MODULE_2__["Fragment"], null, folder_or_plot.subdir && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
      span: 8,
      onClick: function onClick() {
        return setCurrentFolder(folder_or_plot.subdir);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 106,
        columnNumber: 21
      }
    }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["Icon"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 107,
        columnNumber: 23
      }
    }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledA"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 108,
        columnNumber: 23
      }
    }, folder_or_plot.subdir)));
  })), data_get_by_mount.isLoading && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    style: {
      width: '100%',
      display: 'flex',
      justifyContent: 'center',
      height: '100%',
      alignItems: 'center'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 117,
      columnNumber: 11
    }
  }, __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_12__["Spinner"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 118,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 122,
      columnNumber: 11
    }
  }, !data_get_by_mount.isLoading && folders_or_plots.map(function (folder_or_plot) {
    return __jsx(react__WEBPACK_IMPORTED_MODULE_2__["Fragment"], null, folder_or_plot.name && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
      span: 16,
      onClick: function onClick() {
        return setPlot(folder_or_plot);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 127,
        columnNumber: 21
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Button"], {
      block: true,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 128,
        columnNumber: 23
      }
    }, folder_or_plot.name)));
  }))));
};

_s(OverlayWithAnotherPlot, "22XQJ3uLL7mKQo1ldpFWyLcRAOs=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_4__["useRouter"], _hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__["useRequest"]];
});

_c = OverlayWithAnotherPlot;

var _c;

$RefreshReg$(_c, "OverlayWithAnotherPlot");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9vdmVybGF5V2l0aEFub3RoZXJQbG90L2luZGV4LnRzeCJdLCJuYW1lcyI6WyJPdmVybGF5V2l0aEFub3RoZXJQbG90IiwidmlzaWJsZSIsInNldE9wZW5PdmVybGF5V2l0aEFub3RoZXJQbG90TW9kYWwiLCJSZWFjdCIsImZvbGRlcl9wYXRoIiwibmFtZSIsIm92ZXJsYWlkUGxvdHMiLCJzZXRPdmVybGFpZFBsb3RzIiwiZm9sZGVycyIsInNldEZvbGRlcnMiLCJjdXJyZW50Rm9sZGVyIiwic2V0Q3VycmVudEZvbGRlciIsInBsb3QiLCJzZXRQbG90IiwiaGVpZ2h0Iiwic2V0SGVpZ2h0Iiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJzdG9yZSIsInVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4iLCJwYXJhbXMiLCJkYXRhc2V0X25hbWUiLCJydW5fbnVtYmVyIiwibm90T2xkZXJUaGFuIiwiZm9sZGVyc19wYXRoIiwicGxvdF9uYW1lIiwiYXBpIiwiY2hvb3NlX2FwaSIsImRhdGFfZ2V0X2J5X21vdW50IiwidXNlUmVxdWVzdCIsImNvcHkiLCJpbmRleCIsImluZGV4T2YiLCJyZXN0Iiwic3BsaWNlIiwiam9pbmRlckZvbGRlcnMiLCJqb2luIiwicHVzaCIsImNsZWFuZWRfYXJyYXkiLCJjbGVhbkRlZXAiLCJsZW5ndGgiLCJtb2RhbFJlZiIsImRhdGEiLCJmb2xkZXJzX29yX3Bsb3RzIiwiY2hhbmdlRm9sZGVyUGF0aEJ5QnJlYWRjcnVtYiIsIml0ZW0iLCJmb2xkZXJzX2Zyb21fYnJlYWRjcnVtYiIsInNwbGl0IiwiY2xlYW5lZF9mb2xkZXJzX2FycmF5IiwicGFkZGluZyIsImlzTG9hZGluZyIsIndpZHRoIiwiZmxleCIsIm1hcCIsImZvbGRlcl9vcl9wbG90Iiwic3ViZGlyIiwiZGlzcGxheSIsImp1c3RpZnlDb250ZW50IiwiYWxpZ25JdGVtcyJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUdBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUVBO0FBQ0E7QUFPTyxJQUFNQSxzQkFBc0IsR0FBRyxTQUF6QkEsc0JBQXlCLE9BQWtGO0FBQUE7O0FBQUEsTUFBL0VDLE9BQStFLFFBQS9FQSxPQUErRTtBQUFBLE1BQXRFQyxrQ0FBc0UsUUFBdEVBLGtDQUFzRTs7QUFBQSx3QkFDNUVDLDhDQUFBLENBQTRDO0FBQUVDLGVBQVcsRUFBRSxFQUFmO0FBQW1CQyxRQUFJLEVBQUU7QUFBekIsR0FBNUMsQ0FENEU7QUFBQTtBQUFBLE1BQy9HQyxhQUQrRztBQUFBLE1BQ2hHQyxnQkFEZ0c7O0FBQUEseUJBRXhGSiw4Q0FBQSxDQUF1QyxFQUF2QyxDQUZ3RjtBQUFBO0FBQUEsTUFFL0dLLE9BRitHO0FBQUEsTUFFdEdDLFVBRnNHOztBQUFBLHlCQUc1RU4sOENBQUEsQ0FBbUMsRUFBbkMsQ0FINEU7QUFBQTtBQUFBLE1BRy9HTyxhQUgrRztBQUFBLE1BR2hHQyxnQkFIZ0c7O0FBQUEseUJBSTlGUiw4Q0FBQSxDQUFlLEVBQWYsQ0FKOEY7QUFBQTtBQUFBLE1BSS9HUyxJQUorRztBQUFBLE1BSXpHQyxPQUp5Rzs7QUFBQSx5QkFLMUZWLDhDQUFBLEVBTDBGO0FBQUE7QUFBQSxNQUsvR1csTUFMK0c7QUFBQSxNQUt2R0MsU0FMdUc7O0FBT3RILE1BQU1DLE1BQU0sR0FBR0MsNkRBQVMsRUFBeEI7QUFDQSxNQUFNQyxLQUFpQixHQUFHRixNQUFNLENBQUNFLEtBQWpDOztBQVJzSCwwQkFTaEZmLGdEQUFBLENBQWlCZ0IsK0RBQWpCLENBVGdGO0FBQUEsTUFTOUdDLHlCQVQ4RyxxQkFTOUdBLHlCQVQ4Rzs7QUFXdEgsTUFBTUMsTUFBeUIsR0FBRztBQUNoQ0MsZ0JBQVksRUFBRUosS0FBSyxDQUFDSSxZQURZO0FBRWhDQyxjQUFVLEVBQUVMLEtBQUssQ0FBQ0ssVUFGYztBQUdoQ0MsZ0JBQVksRUFBRUoseUJBSGtCO0FBSWhDSyxnQkFBWSxFQUFFbkIsYUFBYSxDQUFDRixXQUpJO0FBS2hDc0IsYUFBUyxFQUFFcEIsYUFBYSxDQUFDRDtBQUxPLEdBQWxDO0FBUUEsTUFBTXNCLEdBQUcsR0FBR0MsNEVBQVUsQ0FBQ1AsTUFBRCxDQUF0QjtBQUNBLE1BQU1RLGlCQUFpQixHQUFHQyxvRUFBVSxDQUFDSCxHQUFELEVBQ2xDLEVBRGtDLEVBRWxDLENBQUNyQixhQUFhLENBQUNGLFdBQWYsQ0FGa0MsQ0FBcEM7QUFLQUQsaURBQUEsQ0FBZ0IsWUFBTTtBQUNwQixRQUFNNEIsSUFBSSxHQUFHLDZGQUFJdkIsT0FBUCxDQUFWOztBQUNBLFFBQU13QixLQUFLLEdBQUd4QixPQUFPLENBQUN5QixPQUFSLENBQWdCdkIsYUFBaEIsQ0FBZDs7QUFFQSxRQUFJc0IsS0FBSyxJQUFJLENBQWIsRUFBZ0I7QUFDZCxVQUFNRSxJQUFJLEdBQUdILElBQUksQ0FBQ0ksTUFBTCxDQUFZLENBQVosRUFBZUgsS0FBSyxHQUFHLENBQXZCLENBQWI7QUFDQXZCLGdCQUFVLENBQUN5QixJQUFELENBQVY7QUFDQSxVQUFNRSxjQUFjLEdBQUdGLElBQUksQ0FBQ0csSUFBTCxDQUFVLEdBQVYsQ0FBdkI7QUFDQTlCLHNCQUFnQixDQUFDO0FBQUVILG1CQUFXLEVBQUVnQyxjQUFmO0FBQStCL0IsWUFBSSxFQUFFO0FBQXJDLE9BQUQsQ0FBaEI7QUFDRCxLQUxELE1BTUs7QUFDSDBCLFVBQUksQ0FBQ08sSUFBTCxDQUFVNUIsYUFBVixFQURHLENBRUg7QUFDQTtBQUNBOztBQUNBLFVBQU02QixhQUFhLEdBQUdDLGtEQUFTLENBQUNULElBQUQsQ0FBVCxHQUFrQlMsa0RBQVMsQ0FBQ1QsSUFBRCxDQUEzQixHQUFvQyxFQUExRDtBQUNBdEIsZ0JBQVUsQ0FBQzhCLGFBQUQsQ0FBVjs7QUFDQSxVQUFNSCxlQUFjLEdBQUdMLElBQUksQ0FBQ00sSUFBTCxDQUFVLEdBQVYsQ0FBdkI7O0FBQ0EsVUFBSUUsYUFBYSxDQUFDRSxNQUFkLEtBQXlCLENBQTdCLEVBQWdDO0FBQzlCbEMsd0JBQWdCLENBQUM7QUFBRUgscUJBQVcsRUFBRSxFQUFmO0FBQW1CQyxjQUFJLEVBQUU7QUFBekIsU0FBRCxDQUFoQjtBQUNEOztBQUNERSxzQkFBZ0IsQ0FBQztBQUFFSCxtQkFBVyxFQUFFZ0MsZUFBZjtBQUErQi9CLFlBQUksRUFBRTtBQUFyQyxPQUFELENBQWhCO0FBQ0Q7QUFDRixHQXZCRCxFQXVCRyxDQUFDSyxhQUFELENBdkJIO0FBeUJBLE1BQU1nQyxRQUFRLEdBQUd2Qyw0Q0FBQSxDQUFhLElBQWIsQ0FBakI7QUFsRHNILE1Bb0Q5R3dDLElBcEQ4RyxHQW9EckdkLGlCQXBEcUcsQ0FvRDlHYyxJQXBEOEc7QUFxRHRILE1BQU1DLGdCQUFnQixHQUFHRCxJQUFJLEdBQUdBLElBQUksQ0FBQ0EsSUFBUixHQUFlLEVBQTVDOztBQUNBLE1BQU1FLDRCQUE0QixHQUFHLFNBQS9CQSw0QkFBK0IsQ0FBQ0MsSUFBRCxFQUErQjtBQUNsRSxRQUFNQyx1QkFBdUIsR0FBR0QsSUFBSSxDQUFDMUMsV0FBTCxDQUFpQjRDLEtBQWpCLENBQXVCLEdBQXZCLENBQWhDO0FBQ0EsUUFBTUMscUJBQXFCLEdBQUdULGtEQUFTLENBQUNPLHVCQUFELENBQVQsR0FBcUNQLGtEQUFTLENBQUNPLHVCQUFELENBQTlDLEdBQTBFLEVBQXhHO0FBQ0F0QyxjQUFVLENBQUN3QyxxQkFBRCxDQUFWOztBQUNBLFFBQUlBLHFCQUFxQixDQUFDUixNQUF0QixHQUErQixDQUFuQyxFQUFzQztBQUNwQzlCLHNCQUFnQixDQUFDc0MscUJBQXFCLENBQUNBLHFCQUFxQixDQUFDUixNQUF0QixHQUErQixDQUFoQyxDQUF0QixDQUFoQjtBQUNELEtBRkQsTUFHSztBQUNIOUIsc0JBQWdCLENBQUMsRUFBRCxDQUFoQjtBQUNEO0FBQ0YsR0FWRDs7QUFZQSxTQUNFLE1BQUMsMkRBQUQ7QUFDRSxXQUFPLEVBQUVWLE9BRFg7QUFFRSxZQUFRLEVBQUUsb0JBQU07QUFDZEMsd0NBQWtDLENBQUMsS0FBRCxDQUFsQztBQUNBUyxzQkFBZ0IsQ0FBQyxFQUFELENBQWhCO0FBQ0QsS0FMSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBT0UsTUFBQyx3Q0FBRDtBQUFLLFVBQU0sRUFBRSxFQUFiO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdDQUFEO0FBQUssU0FBSyxFQUFFO0FBQUV1QyxhQUFPLEVBQUU7QUFBWCxLQUFaO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLGtGQUFEO0FBQVksZUFBVyxFQUFFNUMsYUFBYSxDQUFDRixXQUF2QztBQUFvRCxnQ0FBNEIsRUFBRXlDLDRCQUFsRjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FERixFQUtJLENBQUNoQixpQkFBaUIsQ0FBQ3NCLFNBQW5CLElBQ0EsTUFBQyx3Q0FBRDtBQUFLLFNBQUssRUFBRTtBQUFFQyxXQUFLLEVBQUUsTUFBVDtBQUFpQkMsVUFBSSxFQUFFO0FBQXZCLEtBQVo7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHVCxnQkFBZ0IsQ0FBQ1UsR0FBakIsQ0FBcUIsVUFBQ0MsY0FBRCxFQUF5QjtBQUM3QyxXQUNFLDREQUNHQSxjQUFjLENBQUNDLE1BQWYsSUFDQyxNQUFDLHdDQUFEO0FBQUssVUFBSSxFQUFFLENBQVg7QUFBYyxhQUFPLEVBQUU7QUFBQSxlQUFNN0MsZ0JBQWdCLENBQUM0QyxjQUFjLENBQUNDLE1BQWhCLENBQXRCO0FBQUEsT0FBdkI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUNFLE1BQUMseUVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQURGLEVBRUUsTUFBQyw0RUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQVVELGNBQWMsQ0FBQ0MsTUFBekIsQ0FGRixDQUZKLENBREY7QUFVRCxHQVhBLENBREgsQ0FOSixFQXFCRzNCLGlCQUFpQixDQUFDc0IsU0FBbEIsSUFDQyxNQUFDLHdDQUFEO0FBQUssU0FBSyxFQUFFO0FBQUVDLFdBQUssRUFBRSxNQUFUO0FBQWlCSyxhQUFPLEVBQUUsTUFBMUI7QUFBa0NDLG9CQUFjLEVBQUUsUUFBbEQ7QUFBNEQ1QyxZQUFNLEVBQUUsTUFBcEU7QUFBNEU2QyxnQkFBVSxFQUFFO0FBQXhGLEtBQVo7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsNEVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBdEJKLEVBMkJJLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLENBQUM5QixpQkFBaUIsQ0FBQ3NCLFNBQW5CLElBQWdDUCxnQkFBZ0IsQ0FBQ1UsR0FBakIsQ0FBcUIsVUFBQ0MsY0FBRCxFQUF5QjtBQUM1RSxXQUNFLDREQUNHQSxjQUFjLENBQUNsRCxJQUFmLElBQ0MsTUFBQyx3Q0FBRDtBQUFLLFVBQUksRUFBRSxFQUFYO0FBQWUsYUFBTyxFQUFFO0FBQUEsZUFBTVEsT0FBTyxDQUFDMEMsY0FBRCxDQUFiO0FBQUEsT0FBeEI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUNFLE1BQUMsMkNBQUQ7QUFBUSxXQUFLLE1BQWI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUFlQSxjQUFjLENBQUNsRCxJQUE5QixDQURGLENBRkosQ0FERjtBQVNELEdBVitCLENBRGxDLENBM0JKLENBUEYsQ0FERjtBQXFERCxDQXZITTs7R0FBTUwsc0I7VUFPSWlCLHFELEVBYVdhLDREOzs7S0FwQmY5QixzQiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4wMGMxMTQ1YWZlYzQ1MDMwMGRiYS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnXHJcbmltcG9ydCBNb2RhbCBmcm9tICdhbnRkL2xpYi9tb2RhbC9Nb2RhbCdcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInXHJcblxyXG5pbXBvcnQgeyBQYXJhbXNGb3JBcGlQcm9wcywgUGxvdG92ZXJsYWlkU2VwYXJhdGVseVByb3BzLCBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnXHJcbmltcG9ydCB7IEljb24sIFN0eWxlZEEgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvc3R5bGVkQ29tcG9uZW50cydcclxuaW1wb3J0IHsgY2hvb3NlX2FwaSB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS91dGlscydcclxuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnXHJcbmltcG9ydCB7IHVzZVJlcXVlc3QgfSBmcm9tICcuLi8uLi9ob29rcy91c2VSZXF1ZXN0J1xyXG5pbXBvcnQgeyBCdXR0b24sIENvbCwgUm93IH0gZnJvbSAnYW50ZCdcclxuaW1wb3J0IHsgRm9sZGVyUGF0aCB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9jb250ZW50L2ZvbGRlclBhdGgnXHJcbmltcG9ydCB7IFBhcnNlZFVybFF1ZXJ5SW5wdXQgfSBmcm9tICdxdWVyeXN0cmluZydcclxuaW1wb3J0IGNsZWFuRGVlcCBmcm9tICdjbGVhbi1kZWVwJ1xyXG5pbXBvcnQgeyBTcGlubmVyIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvc3R5bGVkQ29tcG9uZW50cydcclxuXHJcbmludGVyZmFjZSBPdmVybGF5V2l0aEFub3RoZXJQbG90UHJvcHMge1xyXG4gIHZpc2libGU6IGJvb2xlYW47XHJcbiAgc2V0T3Blbk92ZXJsYXlXaXRoQW5vdGhlclBsb3RNb2RhbDogYW55XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBPdmVybGF5V2l0aEFub3RoZXJQbG90ID0gKHsgdmlzaWJsZSwgc2V0T3Blbk92ZXJsYXlXaXRoQW5vdGhlclBsb3RNb2RhbCB9OiBPdmVybGF5V2l0aEFub3RoZXJQbG90UHJvcHMpID0+IHtcclxuICBjb25zdCBbb3ZlcmxhaWRQbG90cywgc2V0T3ZlcmxhaWRQbG90c10gPSBSZWFjdC51c2VTdGF0ZTxQbG90b3ZlcmxhaWRTZXBhcmF0ZWx5UHJvcHM+KHsgZm9sZGVyX3BhdGg6ICcnLCBuYW1lOiAnJyB9KVxyXG4gIGNvbnN0IFtmb2xkZXJzLCBzZXRGb2xkZXJzXSA9IFJlYWN0LnVzZVN0YXRlPChzdHJpbmcgfCB1bmRlZmluZWQpW10+KFtdKVxyXG4gIGNvbnN0IFtjdXJyZW50Rm9sZGVyLCBzZXRDdXJyZW50Rm9sZGVyXSA9IFJlYWN0LnVzZVN0YXRlPHN0cmluZyB8IHVuZGVmaW5lZD4oJycpXHJcbiAgY29uc3QgW3Bsb3QsIHNldFBsb3RdID0gUmVhY3QudXNlU3RhdGUoe30pXHJcbiAgY29uc3QgW2hlaWdodCwgc2V0SGVpZ2h0XSA9IFJlYWN0LnVzZVN0YXRlKClcclxuXHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcbiAgY29uc3QgeyB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIH0gPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKVxyXG5cclxuICBjb25zdCBwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzID0ge1xyXG4gICAgZGF0YXNldF9uYW1lOiBxdWVyeS5kYXRhc2V0X25hbWUgYXMgc3RyaW5nLFxyXG4gICAgcnVuX251bWJlcjogcXVlcnkucnVuX251bWJlciBhcyBzdHJpbmcsXHJcbiAgICBub3RPbGRlclRoYW46IHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4sXHJcbiAgICBmb2xkZXJzX3BhdGg6IG92ZXJsYWlkUGxvdHMuZm9sZGVyX3BhdGgsXHJcbiAgICBwbG90X25hbWU6IG92ZXJsYWlkUGxvdHMubmFtZVxyXG4gIH1cclxuXHJcbiAgY29uc3QgYXBpID0gY2hvb3NlX2FwaShwYXJhbXMpXHJcbiAgY29uc3QgZGF0YV9nZXRfYnlfbW91bnQgPSB1c2VSZXF1ZXN0KGFwaSxcclxuICAgIHt9LFxyXG4gICAgW292ZXJsYWlkUGxvdHMuZm9sZGVyX3BhdGhdXHJcbiAgKTtcclxuXHJcbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGNvbnN0IGNvcHkgPSBbLi4uZm9sZGVyc11cclxuICAgIGNvbnN0IGluZGV4ID0gZm9sZGVycy5pbmRleE9mKGN1cnJlbnRGb2xkZXIpXHJcblxyXG4gICAgaWYgKGluZGV4ID49IDApIHtcclxuICAgICAgY29uc3QgcmVzdCA9IGNvcHkuc3BsaWNlKDAsIGluZGV4ICsgMSlcclxuICAgICAgc2V0Rm9sZGVycyhyZXN0KVxyXG4gICAgICBjb25zdCBqb2luZGVyRm9sZGVycyA9IHJlc3Quam9pbignLycpXHJcbiAgICAgIHNldE92ZXJsYWlkUGxvdHMoeyBmb2xkZXJfcGF0aDogam9pbmRlckZvbGRlcnMsIG5hbWU6ICcnIH0pXHJcbiAgICB9XHJcbiAgICBlbHNlIHtcclxuICAgICAgY29weS5wdXNoKGN1cnJlbnRGb2xkZXIpXHJcbiAgICAgIC8vd2UncmUgY2xlYW5pbmcgY29weSBhcnJheSwgYmVjYXVzZSB3ZSB3YW50IHRvIGRlbGV0ZSBlbXB0eSBzdHJpbmcuIFxyXG4gICAgICAvLyBXZSBuZWVkIHRvIHJlbW92ZSBpdCBiZWNhdXNlIHdoZW4gd2UncmUgam9pbmluZyBhcnJheSB3aXRoIGVtcHR5IHN0cmluZyBcclxuICAgICAgLy8gd2UncmUgZ2V0dGluZyBhIHN0cmluZyB3aXRoICcvJyBpbiB0aGUgYmVnaW5uaW5nLlxyXG4gICAgICBjb25zdCBjbGVhbmVkX2FycmF5ID0gY2xlYW5EZWVwKGNvcHkpID8gY2xlYW5EZWVwKGNvcHkpIDogW11cclxuICAgICAgc2V0Rm9sZGVycyhjbGVhbmVkX2FycmF5KVxyXG4gICAgICBjb25zdCBqb2luZGVyRm9sZGVycyA9IGNvcHkuam9pbignLycpXHJcbiAgICAgIGlmIChjbGVhbmVkX2FycmF5Lmxlbmd0aCA9PT0gMCkge1xyXG4gICAgICAgIHNldE92ZXJsYWlkUGxvdHMoeyBmb2xkZXJfcGF0aDogJycsIG5hbWU6ICcnIH0pXHJcbiAgICAgIH1cclxuICAgICAgc2V0T3ZlcmxhaWRQbG90cyh7IGZvbGRlcl9wYXRoOiBqb2luZGVyRm9sZGVycywgbmFtZTogJycgfSlcclxuICAgIH1cclxuICB9LCBbY3VycmVudEZvbGRlcl0pXHJcblxyXG4gIGNvbnN0IG1vZGFsUmVmID0gUmVhY3QudXNlUmVmKG51bGwpO1xyXG5cclxuICBjb25zdCB7IGRhdGEgfSA9IGRhdGFfZ2V0X2J5X21vdW50XHJcbiAgY29uc3QgZm9sZGVyc19vcl9wbG90cyA9IGRhdGEgPyBkYXRhLmRhdGEgOiBbXVxyXG4gIGNvbnN0IGNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWIgPSAoaXRlbTogUGFyc2VkVXJsUXVlcnlJbnB1dCkgPT4ge1xyXG4gICAgY29uc3QgZm9sZGVyc19mcm9tX2JyZWFkY3J1bWIgPSBpdGVtLmZvbGRlcl9wYXRoLnNwbGl0KCcvJykgXHJcbiAgICBjb25zdCBjbGVhbmVkX2ZvbGRlcnNfYXJyYXkgPSBjbGVhbkRlZXAoZm9sZGVyc19mcm9tX2JyZWFkY3J1bWIpID8gY2xlYW5EZWVwKGZvbGRlcnNfZnJvbV9icmVhZGNydW1iKSA6IFtdXHJcbiAgICBzZXRGb2xkZXJzKGNsZWFuZWRfZm9sZGVyc19hcnJheSlcclxuICAgIGlmIChjbGVhbmVkX2ZvbGRlcnNfYXJyYXkubGVuZ3RoID4gMCkge1xyXG4gICAgICBzZXRDdXJyZW50Rm9sZGVyKGNsZWFuZWRfZm9sZGVyc19hcnJheVtjbGVhbmVkX2ZvbGRlcnNfYXJyYXkubGVuZ3RoIC0gMV0pXHJcbiAgICB9XHJcbiAgICBlbHNlIHtcclxuICAgICAgc2V0Q3VycmVudEZvbGRlcignJylcclxuICAgIH1cclxuICB9XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8TW9kYWxcclxuICAgICAgdmlzaWJsZT17dmlzaWJsZX1cclxuICAgICAgb25DYW5jZWw9eygpID0+IHtcclxuICAgICAgICBzZXRPcGVuT3ZlcmxheVdpdGhBbm90aGVyUGxvdE1vZGFsKGZhbHNlKVxyXG4gICAgICAgIHNldEN1cnJlbnRGb2xkZXIoJycpXHJcbiAgICAgIH19XHJcbiAgICA+XHJcbiAgICAgIDxSb3cgZ3V0dGVyPXsxNn0gPlxyXG4gICAgICAgIDxDb2wgc3R5bGU9e3sgcGFkZGluZzogOCB9fT5cclxuICAgICAgICAgIDxGb2xkZXJQYXRoIGZvbGRlcl9wYXRoPXtvdmVybGFpZFBsb3RzLmZvbGRlcl9wYXRofSBjaGFuZ2VGb2xkZXJQYXRoQnlCcmVhZGNydW1iPXtjaGFuZ2VGb2xkZXJQYXRoQnlCcmVhZGNydW1ifSAvPlxyXG4gICAgICAgIDwvQ29sPlxyXG4gICAgICAgIHtcclxuICAgICAgICAgICFkYXRhX2dldF9ieV9tb3VudC5pc0xvYWRpbmcgJiZcclxuICAgICAgICAgIDxSb3cgc3R5bGU9e3sgd2lkdGg6ICcxMDAlJywgZmxleDogJzEgMSBhdXRvJyB9fT5cclxuICAgICAgICAgICAge2ZvbGRlcnNfb3JfcGxvdHMubWFwKChmb2xkZXJfb3JfcGxvdDogYW55KSA9PiB7XHJcbiAgICAgICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgICAgIDw+XHJcbiAgICAgICAgICAgICAgICAgIHtmb2xkZXJfb3JfcGxvdC5zdWJkaXIgJiZcclxuICAgICAgICAgICAgICAgICAgICA8Q29sIHNwYW49ezh9IG9uQ2xpY2s9eygpID0+IHNldEN1cnJlbnRGb2xkZXIoZm9sZGVyX29yX3Bsb3Quc3ViZGlyKX0+XHJcbiAgICAgICAgICAgICAgICAgICAgICA8SWNvbiAvPlxyXG4gICAgICAgICAgICAgICAgICAgICAgPFN0eWxlZEE+e2ZvbGRlcl9vcl9wbG90LnN1YmRpcn08L1N0eWxlZEE+XHJcbiAgICAgICAgICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIDwvPlxyXG4gICAgICAgICAgICAgIClcclxuICAgICAgICAgICAgfSl9XHJcbiAgICAgICAgICA8L1Jvdz5cclxuICAgICAgICB9XHJcbiAgICAgICAge2RhdGFfZ2V0X2J5X21vdW50LmlzTG9hZGluZyAmJlxyXG4gICAgICAgICAgPFJvdyBzdHlsZT17eyB3aWR0aDogJzEwMCUnLCBkaXNwbGF5OiAnZmxleCcsIGp1c3RpZnlDb250ZW50OiAnY2VudGVyJywgaGVpZ2h0OiAnMTAwJScsIGFsaWduSXRlbXM6ICdjZW50ZXInIH19PlxyXG4gICAgICAgICAgICA8U3Bpbm5lciAvPlxyXG4gICAgICAgICAgPC9Sb3c+XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHtcclxuICAgICAgICAgIDxSb3c+e1xyXG4gICAgICAgICAgICAhZGF0YV9nZXRfYnlfbW91bnQuaXNMb2FkaW5nICYmIGZvbGRlcnNfb3JfcGxvdHMubWFwKChmb2xkZXJfb3JfcGxvdDogYW55KSA9PiB7XHJcbiAgICAgICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgICAgIDw+XHJcbiAgICAgICAgICAgICAgICAgIHtmb2xkZXJfb3JfcGxvdC5uYW1lICYmXHJcbiAgICAgICAgICAgICAgICAgICAgPENvbCBzcGFuPXsxNn0gb25DbGljaz17KCkgPT4gc2V0UGxvdChmb2xkZXJfb3JfcGxvdCl9PlxyXG4gICAgICAgICAgICAgICAgICAgICAgPEJ1dHRvbiBibG9jaz57Zm9sZGVyX29yX3Bsb3QubmFtZX08L0J1dHRvbj5cclxuICAgICAgICAgICAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgPC8+XHJcbiAgICAgICAgICAgICAgKVxyXG4gICAgICAgICAgICB9KVxyXG4gICAgICAgICAgfVxyXG4gICAgICAgICAgPC9Sb3c+XHJcbiAgICAgICAgfVxyXG4gICAgICA8L1Jvdz5cclxuICAgIDwvTW9kYWw+XHJcbiAgKVxyXG59Il0sInNvdXJjZVJvb3QiOiIifQ==