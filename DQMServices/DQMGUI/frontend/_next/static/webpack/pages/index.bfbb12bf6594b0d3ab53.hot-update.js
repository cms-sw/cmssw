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

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_2__["useState"](''),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState3, 2),
      folderPath = _React$useState4[0],
      setFolderPath = _React$useState4[1];

  var _React$useState5 = react__WEBPACK_IMPORTED_MODULE_2__["useState"]([]),
      _React$useState6 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState5, 2),
      folders = _React$useState6[0],
      setFolders = _React$useState6[1];

  var _React$useState7 = react__WEBPACK_IMPORTED_MODULE_2__["useState"](''),
      _React$useState8 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState7, 2),
      currentFolder = _React$useState8[0],
      setCurrentFolder = _React$useState8[1];

  var _React$useState9 = react__WEBPACK_IMPORTED_MODULE_2__["useState"]({}),
      _React$useState10 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState9, 2),
      plot = _React$useState10[0],
      setPlot = _React$useState10[1];

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
      setFolders(folders.filter(function (folder) {
        return folder !== folders[index];
      }));
    } else {
      copy.push(currentFolder);
      setFolders(copy);
    }
  }, [currentFolder]);
  var data = data_get_by_mount.data;
  var folders_or_plots = data ? data.data : [];

  var changeFolderPathByBreadcrumb = function changeFolderPathByBreadcrumb(item) {
    var folders_from_breadcrumb = item.folder_path.split('/');
    setCurrentFolder(folders_from_breadcrumb[folders_from_breadcrumb.length - 1]);
    setFolders(folders_from_breadcrumb);
  };

  console.log(currentFolder);
  return __jsx(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3___default.a, {
    visible: visible,
    onCancel: function onCancel() {
      setOpenOverlayWithAnotherPlotModal(false);
      setFolderPath([]);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 68,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    gutter: 16,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 75,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
    style: {
      padding: 8
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 76,
      columnNumber: 9
    }
  }, __jsx(_containers_display_content_folderPath__WEBPACK_IMPORTED_MODULE_10__["FolderPath"], {
    folder_path: overlaidPlots.folder_path,
    changeFolderPathByBreadcrumb: changeFolderPathByBreadcrumb,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 77,
      columnNumber: 11
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    style: {
      width: '100%'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 79,
      columnNumber: 9
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
        lineNumber: 85,
        columnNumber: 21
      }
    }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["Icon"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 86,
        columnNumber: 23
      }
    }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledA"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 87,
        columnNumber: 23
      }
    }, folder_or_plot.subdir)));
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    style: {
      width: '100%'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 95,
      columnNumber: 9
    }
  }, folders_or_plots.map(function (folder_or_plot) {
    return __jsx(react__WEBPACK_IMPORTED_MODULE_2__["Fragment"], null, folder_or_plot.name && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
      span: 8,
      onClick: function onClick() {
        return setPlot(folder_or_plot);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 101,
        columnNumber: 21
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Button"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 102,
        columnNumber: 23
      }
    }, folder_or_plot.name)));
  }))));
};

_s(OverlayWithAnotherPlot, "asT887bGchGBdSh2Mw26i8xI0hk=", false, function () {
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9vdmVybGF5V2l0aEFub3RoZXJQbG90L2luZGV4LnRzeCJdLCJuYW1lcyI6WyJPdmVybGF5V2l0aEFub3RoZXJQbG90IiwidmlzaWJsZSIsInNldE9wZW5PdmVybGF5V2l0aEFub3RoZXJQbG90TW9kYWwiLCJSZWFjdCIsImZvbGRlcl9wYXRoIiwibmFtZSIsIm92ZXJsYWlkUGxvdHMiLCJzZXRPdmVybGFpZFBsb3RzIiwiZm9sZGVyUGF0aCIsInNldEZvbGRlclBhdGgiLCJmb2xkZXJzIiwic2V0Rm9sZGVycyIsImN1cnJlbnRGb2xkZXIiLCJzZXRDdXJyZW50Rm9sZGVyIiwicGxvdCIsInNldFBsb3QiLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJxdWVyeSIsInN0b3JlIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsInBhcmFtcyIsImRhdGFzZXRfbmFtZSIsInJ1bl9udW1iZXIiLCJub3RPbGRlclRoYW4iLCJmb2xkZXJzX3BhdGgiLCJwbG90X25hbWUiLCJhcGkiLCJjaG9vc2VfYXBpIiwiZGF0YV9nZXRfYnlfbW91bnQiLCJ1c2VSZXF1ZXN0IiwiY29weSIsImluZGV4IiwiaW5kZXhPZiIsImZpbHRlciIsImZvbGRlciIsInB1c2giLCJkYXRhIiwiZm9sZGVyc19vcl9wbG90cyIsImNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWIiLCJpdGVtIiwiZm9sZGVyc19mcm9tX2JyZWFkY3J1bWIiLCJzcGxpdCIsImxlbmd0aCIsImNvbnNvbGUiLCJsb2ciLCJwYWRkaW5nIiwid2lkdGgiLCJtYXAiLCJmb2xkZXJfb3JfcGxvdCIsInN1YmRpciJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUdBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQVFPLElBQU1BLHNCQUFzQixHQUFHLFNBQXpCQSxzQkFBeUIsT0FBa0Y7QUFBQTs7QUFB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WEQsQ0FGSixDQUpGLEVBb0JFLE1BQUMsd0NBQUQ7QUFBSyxTQUFLLEVBQUU7QUFBRUgsV0FBSyxFQUFFO0FBQVQsS0FBWjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBRUlULGdCQUFnQixDQUFDVSxHQUFqQixDQUFxQixVQUFDQyxjQUFELEVBQXlCO0FBQzVDLFdBQ0UsNERBQ0dBLGNBQWMsQ0FBQzVDLElBQWYsSUFDQyxNQUFDLHdDQUFEO0FBQUssVUFBSSxFQUFFLENBQVg7QUFBYyxhQUFPLEVBQUU7QUFBQSxlQUFNVSxPQUFPLENBQUNrQyxjQUFELENBQWI7QUFBQSxPQUF2QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0UsTUFBQywyQ0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQVVBLGNBQWMsQ0FBQzVDLElBQXpCLENBREYsQ0FGSixDQURGO0FBU0QsR0FWRCxDQUZKLENBcEJGLENBUEYsQ0FERjtBQThDRCxDQTlGTTs7R0FBTUwsc0I7VUFPSWlCLHFELEVBYVdhLDREOzs7S0FwQmY5QixzQiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC5iZmJiMTJiZjY1OTRiMGQzYWI1My5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnXHJcbmltcG9ydCBNb2RhbCBmcm9tICdhbnRkL2xpYi9tb2RhbC9Nb2RhbCdcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInXHJcblxyXG5pbXBvcnQgeyBQYXJhbXNGb3JBcGlQcm9wcywgUGxvdG92ZXJsYWlkU2VwYXJhdGVseVByb3BzLCBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnXHJcbmltcG9ydCB7IEljb24sIFN0eWxlZEEgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvc3R5bGVkQ29tcG9uZW50cydcclxuaW1wb3J0IHsgY2hvb3NlX2FwaSB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS91dGlscydcclxuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnXHJcbmltcG9ydCB7IHVzZVJlcXVlc3QgfSBmcm9tICcuLi8uLi9ob29rcy91c2VSZXF1ZXN0J1xyXG5pbXBvcnQgeyBCdXR0b24sIENvbCwgUm93IH0gZnJvbSAnYW50ZCdcclxuaW1wb3J0IHsgRm9sZGVyUGF0aCB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9jb250ZW50L2ZvbGRlclBhdGgnXHJcbmltcG9ydCB7IFBhcnNlZFVybFF1ZXJ5SW5wdXQgfSBmcm9tICdxdWVyeXN0cmluZydcclxuXHJcbmludGVyZmFjZSBPdmVybGF5V2l0aEFub3RoZXJQbG90UHJvcHMge1xyXG4gIHZpc2libGU6IGJvb2xlYW47XHJcbiAgc2V0T3Blbk92ZXJsYXlXaXRoQW5vdGhlclBsb3RNb2RhbDogYW55XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBPdmVybGF5V2l0aEFub3RoZXJQbG90ID0gKHsgdmlzaWJsZSwgc2V0T3Blbk92ZXJsYXlXaXRoQW5vdGhlclBsb3RNb2RhbCB9OiBPdmVybGF5V2l0aEFub3RoZXJQbG90UHJvcHMpID0+IHtcclxuICBjb25zdCBbb3ZlcmxhaWRQbG90cywgc2V0T3ZlcmxhaWRQbG90c10gPSBSZWFjdC51c2VTdGF0ZTxQbG90b3ZlcmxhaWRTZXBhcmF0ZWx5UHJvcHM+KHsgZm9sZGVyX3BhdGg6ICcnLCBuYW1lOiAnJyB9KVxyXG4gIGNvbnN0IFtmb2xkZXJQYXRoLCBzZXRGb2xkZXJQYXRoXSA9IFJlYWN0LnVzZVN0YXRlPHN0cmluZ1tdPignJylcclxuICBjb25zdCBbZm9sZGVycywgc2V0Rm9sZGVyc10gPSBSZWFjdC51c2VTdGF0ZTxzdHJpbmdbXT4oW10pXHJcbiAgY29uc3QgW2N1cnJlbnRGb2xkZXIsIHNldEN1cnJlbnRGb2xkZXJdID0gUmVhY3QudXNlU3RhdGUoJycpXHJcbiAgY29uc3QgW3Bsb3QsIHNldFBsb3RdID0gUmVhY3QudXNlU3RhdGUoe30pXHJcblxyXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xyXG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xyXG4gIGNvbnN0IHsgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSlcclxuXHJcbiAgY29uc3QgcGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcyA9IHtcclxuICAgIGRhdGFzZXRfbmFtZTogcXVlcnkuZGF0YXNldF9uYW1lIGFzIHN0cmluZyxcclxuICAgIHJ1bl9udW1iZXI6IHF1ZXJ5LnJ1bl9udW1iZXIgYXMgc3RyaW5nLFxyXG4gICAgbm90T2xkZXJUaGFuOiB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxyXG4gICAgZm9sZGVyc19wYXRoOiBvdmVybGFpZFBsb3RzLmZvbGRlcl9wYXRoLFxyXG4gICAgcGxvdF9uYW1lOiBvdmVybGFpZFBsb3RzLm5hbWVcclxuICB9XHJcblxyXG4gIGNvbnN0IGFwaSA9IGNob29zZV9hcGkocGFyYW1zKVxyXG4gIGNvbnN0IGRhdGFfZ2V0X2J5X21vdW50ID0gdXNlUmVxdWVzdChhcGksXHJcbiAgICB7fSxcclxuICAgIFtvdmVybGFpZFBsb3RzLmZvbGRlcl9wYXRoXVxyXG4gICk7XHJcblxyXG4gIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBjb25zdCBjb3B5ID0gWy4uLmZvbGRlcnNdXHJcbiAgICBjb25zdCBpbmRleCA9IGZvbGRlcnMuaW5kZXhPZihjdXJyZW50Rm9sZGVyKVxyXG5cclxuICAgIGlmIChpbmRleCA+PSAwKSB7XHJcbiAgICAgIHNldEZvbGRlcnMoZm9sZGVycy5maWx0ZXIoKGZvbGRlcjogc3RyaW5nKSA9PiBmb2xkZXIgIT09IGZvbGRlcnNbaW5kZXhdKSlcclxuICAgIH1cclxuICAgIGVsc2Uge1xyXG4gICAgICBjb3B5LnB1c2goY3VycmVudEZvbGRlcilcclxuICAgICAgc2V0Rm9sZGVycyhjb3B5KVxyXG4gICAgfVxyXG4gIH0sIFtjdXJyZW50Rm9sZGVyXSlcclxuXHJcbiAgY29uc3QgeyBkYXRhIH0gPSBkYXRhX2dldF9ieV9tb3VudFxyXG4gIGNvbnN0IGZvbGRlcnNfb3JfcGxvdHMgPSBkYXRhID8gZGF0YS5kYXRhIDogW11cclxuXHJcbiAgY29uc3QgY2hhbmdlRm9sZGVyUGF0aEJ5QnJlYWRjcnVtYiA9IChpdGVtKSA9PiB7XHJcbiAgICBjb25zdCBmb2xkZXJzX2Zyb21fYnJlYWRjcnVtYiA9IGl0ZW0uZm9sZGVyX3BhdGguc3BsaXQoJy8nKVxyXG4gICAgc2V0Q3VycmVudEZvbGRlcihmb2xkZXJzX2Zyb21fYnJlYWRjcnVtYltmb2xkZXJzX2Zyb21fYnJlYWRjcnVtYi5sZW5ndGggLSAxXSlcclxuICAgIHNldEZvbGRlcnMoZm9sZGVyc19mcm9tX2JyZWFkY3J1bWIpXHJcbiAgfVxyXG5cclxuICBjb25zb2xlLmxvZyhjdXJyZW50Rm9sZGVyKVxyXG4gIHJldHVybiAoXHJcbiAgICA8TW9kYWxcclxuICAgICAgdmlzaWJsZT17dmlzaWJsZX1cclxuICAgICAgb25DYW5jZWw9eygpID0+IHtcclxuICAgICAgICBzZXRPcGVuT3ZlcmxheVdpdGhBbm90aGVyUGxvdE1vZGFsKGZhbHNlKVxyXG4gICAgICAgIHNldEZvbGRlclBhdGgoW10pXHJcbiAgICAgIH19XHJcbiAgICA+XHJcbiAgICAgIDxSb3cgZ3V0dGVyPXsxNn0+XHJcbiAgICAgICAgPENvbCBzdHlsZT17eyBwYWRkaW5nOiA4IH19PlxyXG4gICAgICAgICAgPEZvbGRlclBhdGggZm9sZGVyX3BhdGg9e292ZXJsYWlkUGxvdHMuZm9sZGVyX3BhdGh9IGNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWI9e2NoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWJ9IC8+XHJcbiAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgPFJvdyBzdHlsZT17eyB3aWR0aDogJzEwMCUnIH19PlxyXG4gICAgICAgICAge1xyXG4gICAgICAgICAgICBmb2xkZXJzX29yX3Bsb3RzLm1hcCgoZm9sZGVyX29yX3Bsb3Q6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICA8PlxyXG4gICAgICAgICAgICAgICAgICB7Zm9sZGVyX29yX3Bsb3Quc3ViZGlyICYmXHJcbiAgICAgICAgICAgICAgICAgICAgPENvbCBzcGFuPXs4fSBvbkNsaWNrPXsoKSA9PiBzZXRDdXJyZW50Rm9sZGVyKGZvbGRlcl9vcl9wbG90LnN1YmRpcil9PlxyXG4gICAgICAgICAgICAgICAgICAgICAgPEljb24gLz5cclxuICAgICAgICAgICAgICAgICAgICAgIDxTdHlsZWRBPntmb2xkZXJfb3JfcGxvdC5zdWJkaXJ9PC9TdHlsZWRBPlxyXG4gICAgICAgICAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICA8Lz5cclxuICAgICAgICAgICAgICApXHJcbiAgICAgICAgICAgIH0pXHJcbiAgICAgICAgICB9XHJcbiAgICAgICAgPC9Sb3c+XHJcbiAgICAgICAgPFJvdyBzdHlsZT17eyB3aWR0aDogJzEwMCUnIH19PlxyXG4gICAgICAgICAge1xyXG4gICAgICAgICAgICBmb2xkZXJzX29yX3Bsb3RzLm1hcCgoZm9sZGVyX29yX3Bsb3Q6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICA8PlxyXG4gICAgICAgICAgICAgICAgICB7Zm9sZGVyX29yX3Bsb3QubmFtZSAmJlxyXG4gICAgICAgICAgICAgICAgICAgIDxDb2wgc3Bhbj17OH0gb25DbGljaz17KCkgPT4gc2V0UGxvdChmb2xkZXJfb3JfcGxvdCl9PlxyXG4gICAgICAgICAgICAgICAgICAgICAgPEJ1dHRvbiA+e2ZvbGRlcl9vcl9wbG90Lm5hbWV9PC9CdXR0b24+XHJcbiAgICAgICAgICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIDwvPlxyXG4gICAgICAgICAgICAgIClcclxuICAgICAgICAgICAgfSlcclxuICAgICAgICAgIH1cclxuICAgICAgICA8L1Jvdz5cclxuICAgICAgPC9Sb3c+XHJcbiAgICA8L01vZGFsPlxyXG4gIClcclxufSJdLCJzb3VyY2VSb290IjoiIn0=