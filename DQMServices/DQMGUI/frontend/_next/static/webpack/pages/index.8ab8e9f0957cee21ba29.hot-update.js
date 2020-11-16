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

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_2__["useState"]([]),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState, 2),
      data = _React$useState2[0],
      setData = _React$useState2[1];

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_2__["useState"]({
    folder_path: '',
    name: ''
  }),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState3, 2),
      overlaidPlots = _React$useState4[0],
      setOverlaidPlots = _React$useState4[1];

  var _React$useState5 = react__WEBPACK_IMPORTED_MODULE_2__["useState"]([]),
      _React$useState6 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState5, 2),
      folderPath = _React$useState6[0],
      setFolderPath = _React$useState6[1];

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
    if (data_get_by_mount && data_get_by_mount.data) {
      setData(data_get_by_mount.data.data);
    }

    console.log();
  }, [data_get_by_mount.data]);
  react__WEBPACK_IMPORTED_MODULE_2__["useEffect"](function () {
    var copy = Object(_babel_runtime_helpers_esm_toConsumableArray__WEBPACK_IMPORTED_MODULE_0__["default"])(folderPath);

    var newItemIndexInFolderPath = copy.indexOf(currentFolder);
    console.log(newItemIndexInFolderPath, newItemIndexInFolderPath, currentFolder);

    if (newItemIndexInFolderPath > -1) {
      var howManyItemsNeedToRemove = copy.length - 1 - newItemIndexInFolderPath;
      console.log(howManyItemsNeedToRemove);
      copy.splice(newItemIndexInFolderPath, howManyItemsNeedToRemove);
    } else {
      copy.push(currentFolder);
    }

    setFolderPath(copy);
    return function () {
      return setFolderPath([]);
    };
  }, [currentFolder, folderPath[folderPath.length - 1]]); // when the last folder in path is changed

  react__WEBPACK_IMPORTED_MODULE_2__["useEffect"](function () {
    var joinedFoldersForRequest = folderPath.join('/').substr(1);
    console.log(joinedFoldersForRequest);
    setOverlaidPlots({
      name: '',
      folder_path: joinedFoldersForRequest
    });
  }, [folderPath]);

  var changeFolderPathByBreadcrumb = function changeFolderPathByBreadcrumb(parameters) {
    console.log(parameters);

    if (parameters.folder_path === '/') {
      setOverlaidPlots({
        folder_path: '',
        name: ''
      });
      setFolderPath([]);
      setCurrentFolder('');
    }

    var folders = overlaidPlots.folder_path.split('/');
    setFolderPath(folders);
    setOverlaidPlots(parameters);
  };

  return __jsx(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3___default.a, {
    visible: visible,
    onCancel: function onCancel() {
      setOpenOverlayWithAnotherPlotModal(false);
      setFolderPath([]);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 87,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    gutter: 16,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 94,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
    style: {
      padding: 8
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 95,
      columnNumber: 9
    }
  }, __jsx(_containers_display_content_folderPath__WEBPACK_IMPORTED_MODULE_10__["FolderPath"], {
    folder_path: overlaidPlots.folder_path,
    changeFolderPathByBreadcrumb: changeFolderPathByBreadcrumb,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 96,
      columnNumber: 11
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    style: {
      width: '100%'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 98,
      columnNumber: 9
    }
  }, data.map(function (folder_or_plot) {
    return __jsx(react__WEBPACK_IMPORTED_MODULE_2__["Fragment"], null, folder_or_plot.subdir && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
      span: 8,
      onClick: function onClick() {
        return setCurrentFolder(folder_or_plot.subdir);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 104,
        columnNumber: 21
      }
    }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["Icon"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 105,
        columnNumber: 23
      }
    }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledA"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 106,
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
      lineNumber: 114,
      columnNumber: 9
    }
  }, data.map(function (folder_or_plot) {
    return __jsx(react__WEBPACK_IMPORTED_MODULE_2__["Fragment"], null, folder_or_plot.name && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
      span: 8,
      onClick: function onClick() {
        return setPlot(folder_or_plot);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 120,
        columnNumber: 21
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Button"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 121,
        columnNumber: 23
      }
    }, folder_or_plot.name)));
  }))));
};

_s(OverlayWithAnotherPlot, "2chABmc/YJS6AMKuh3dCNIbNV40=", false, function () {
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9vdmVybGF5V2l0aEFub3RoZXJQbG90L2luZGV4LnRzeCJdLCJuYW1lcyI6WyJPdmVybGF5V2l0aEFub3RoZXJQbG90IiwidmlzaWJsZSIsInNldE9wZW5PdmVybGF5V2l0aEFub3RoZXJQbG90TW9kYWwiLCJSZWFjdCIsImRhdGEiLCJzZXREYXRhIiwiZm9sZGVyX3BhdGgiLCJuYW1lIiwib3ZlcmxhaWRQbG90cyIsInNldE92ZXJsYWlkUGxvdHMiLCJmb2xkZXJQYXRoIiwic2V0Rm9sZGVyUGF0aCIsImN1cnJlbnRGb2xkZXIiLCJzZXRDdXJyZW50Rm9sZGVyIiwicGxvdCIsInNldFBsb3QiLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJxdWVyeSIsInN0b3JlIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsInBhcmFtcyIsImRhdGFzZXRfbmFtZSIsInJ1bl9udW1iZXIiLCJub3RPbGRlclRoYW4iLCJmb2xkZXJzX3BhdGgiLCJwbG90X25hbWUiLCJhcGkiLCJjaG9vc2VfYXBpIiwiZGF0YV9nZXRfYnlfbW91bnQiLCJ1c2VSZXF1ZXN0IiwiY29uc29sZSIsImxvZyIsImNvcHkiLCJuZXdJdGVtSW5kZXhJbkZvbGRlclBhdGgiLCJpbmRleE9mIiwiaG93TWFueUl0ZW1zTmVlZFRvUmVtb3ZlIiwibGVuZ3RoIiwic3BsaWNlIiwicHVzaCIsImpvaW5lZEZvbGRlcnNGb3JSZXF1ZXN0Iiwiam9pbiIsInN1YnN0ciIsImNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWIiLCJwYXJhbWV0ZXJzIiwiZm9sZGVycyIsInNwbGl0IiwicGFkZGluZyIsIndpZHRoIiwibWFwIiwiZm9sZGVyX29yX3Bsb3QiLCJzdWJkaXIiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFHQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFRTyxJQUFNQSxzQkFB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NERBQ0dBLGNBQWMsQ0FBQzNDLElBQWYsSUFDQyxNQUFDLHdDQUFEO0FBQUssVUFBSSxFQUFFLENBQVg7QUFBYyxhQUFPLEVBQUU7QUFBQSxlQUFNUSxPQUFPLENBQUNtQyxjQUFELENBQWI7QUFBQSxPQUF2QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0UsTUFBQywyQ0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQVVBLGNBQWMsQ0FBQzNDLElBQXpCLENBREYsQ0FGSixDQURGO0FBU0QsR0FWRCxDQUZKLENBcEJGLENBUEYsQ0FERjtBQThDRCxDQWpITTs7R0FBTVAsc0I7VUFPSWlCLHFELEVBYVdhLDREOzs7S0FwQmY5QixzQiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC44YWI4ZTlmMDk1N2NlZTIxYmEyOS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnXHJcbmltcG9ydCBNb2RhbCBmcm9tICdhbnRkL2xpYi9tb2RhbC9Nb2RhbCdcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInXHJcblxyXG5pbXBvcnQgeyBQYXJhbXNGb3JBcGlQcm9wcywgUGxvdG92ZXJsYWlkU2VwYXJhdGVseVByb3BzLCBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnXHJcbmltcG9ydCB7IEljb24sIFN0eWxlZEEgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvc3R5bGVkQ29tcG9uZW50cydcclxuaW1wb3J0IHsgY2hvb3NlX2FwaSB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS91dGlscydcclxuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnXHJcbmltcG9ydCB7IHVzZVJlcXVlc3QgfSBmcm9tICcuLi8uLi9ob29rcy91c2VSZXF1ZXN0J1xyXG5pbXBvcnQgeyBCdXR0b24sIENvbCwgUm93IH0gZnJvbSAnYW50ZCdcclxuaW1wb3J0IHsgRm9sZGVyUGF0aCB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9jb250ZW50L2ZvbGRlclBhdGgnXHJcbmltcG9ydCB7IFBhcnNlZFVybFF1ZXJ5SW5wdXQgfSBmcm9tICdxdWVyeXN0cmluZydcclxuXHJcbmludGVyZmFjZSBPdmVybGF5V2l0aEFub3RoZXJQbG90UHJvcHMge1xyXG4gIHZpc2libGU6IGJvb2xlYW47XHJcbiAgc2V0T3Blbk92ZXJsYXlXaXRoQW5vdGhlclBsb3RNb2RhbDogYW55XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBPdmVybGF5V2l0aEFub3RoZXJQbG90ID0gKHsgdmlzaWJsZSwgc2V0T3Blbk92ZXJsYXlXaXRoQW5vdGhlclBsb3RNb2RhbCB9OiBPdmVybGF5V2l0aEFub3RoZXJQbG90UHJvcHMpID0+IHtcclxuICBjb25zdCBbZGF0YSwgc2V0RGF0YV0gPSBSZWFjdC51c2VTdGF0ZTxhbnk+KFtdKVxyXG4gIGNvbnN0IFtvdmVybGFpZFBsb3RzLCBzZXRPdmVybGFpZFBsb3RzXSA9IFJlYWN0LnVzZVN0YXRlPFBsb3RvdmVybGFpZFNlcGFyYXRlbHlQcm9wcz4oeyBmb2xkZXJfcGF0aDogJycsIG5hbWU6ICcnIH0pXHJcbiAgY29uc3QgW2ZvbGRlclBhdGgsIHNldEZvbGRlclBhdGhdID0gUmVhY3QudXNlU3RhdGU8c3RyaW5nW10+KFtdKVxyXG4gIGNvbnN0IFtjdXJyZW50Rm9sZGVyLCBzZXRDdXJyZW50Rm9sZGVyXSA9IFJlYWN0LnVzZVN0YXRlKCcnKVxyXG4gIGNvbnN0IFtwbG90LCBzZXRQbG90XSA9IFJlYWN0LnVzZVN0YXRlKHt9KVxyXG5cclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuICBjb25zdCB7IHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4gfSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpXHJcblxyXG4gIGNvbnN0IHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMgPSB7XHJcbiAgICBkYXRhc2V0X25hbWU6IHF1ZXJ5LmRhdGFzZXRfbmFtZSBhcyBzdHJpbmcsXHJcbiAgICBydW5fbnVtYmVyOiBxdWVyeS5ydW5fbnVtYmVyIGFzIHN0cmluZyxcclxuICAgIG5vdE9sZGVyVGhhbjogdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbixcclxuICAgIGZvbGRlcnNfcGF0aDogb3ZlcmxhaWRQbG90cy5mb2xkZXJfcGF0aCxcclxuICAgIHBsb3RfbmFtZTogb3ZlcmxhaWRQbG90cy5uYW1lXHJcbiAgfVxyXG5cclxuICBjb25zdCBhcGkgPSBjaG9vc2VfYXBpKHBhcmFtcylcclxuICBjb25zdCBkYXRhX2dldF9ieV9tb3VudCA9IHVzZVJlcXVlc3QoYXBpLFxyXG4gICAge30sXHJcbiAgICBbb3ZlcmxhaWRQbG90cy5mb2xkZXJfcGF0aF1cclxuICApO1xyXG5cclxuICBSZWFjdC51c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgaWYgKGRhdGFfZ2V0X2J5X21vdW50ICYmIGRhdGFfZ2V0X2J5X21vdW50LmRhdGEpIHtcclxuICAgICAgc2V0RGF0YShkYXRhX2dldF9ieV9tb3VudC5kYXRhLmRhdGEpXHJcbiAgICB9XHJcbiAgICBjb25zb2xlLmxvZygpXHJcbiAgfSwgW2RhdGFfZ2V0X2J5X21vdW50LmRhdGFdKVxyXG5cclxuICBSZWFjdC51c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgY29uc3QgY29weSA9IFsuLi5mb2xkZXJQYXRoXVxyXG4gICAgY29uc3QgbmV3SXRlbUluZGV4SW5Gb2xkZXJQYXRoID0gY29weS5pbmRleE9mKGN1cnJlbnRGb2xkZXIpXHJcbiAgICBjb25zb2xlLmxvZyhuZXdJdGVtSW5kZXhJbkZvbGRlclBhdGgsIG5ld0l0ZW1JbmRleEluRm9sZGVyUGF0aCwgY3VycmVudEZvbGRlcilcclxuICAgIGlmIChuZXdJdGVtSW5kZXhJbkZvbGRlclBhdGggPiAtMSkge1xyXG4gICAgICBjb25zdCBob3dNYW55SXRlbXNOZWVkVG9SZW1vdmUgPSAoY29weS5sZW5ndGggLSAxKSAtIG5ld0l0ZW1JbmRleEluRm9sZGVyUGF0aFxyXG4gICAgICBjb25zb2xlLmxvZyhob3dNYW55SXRlbXNOZWVkVG9SZW1vdmUpXHJcbiAgICAgIGNvcHkuc3BsaWNlKG5ld0l0ZW1JbmRleEluRm9sZGVyUGF0aCwgaG93TWFueUl0ZW1zTmVlZFRvUmVtb3ZlKVxyXG4gICAgfVxyXG4gICAgZWxzZSB7XHJcbiAgICAgIGNvcHkucHVzaChjdXJyZW50Rm9sZGVyKVxyXG4gICAgfVxyXG4gICAgc2V0Rm9sZGVyUGF0aChjb3B5KVxyXG4gICAgcmV0dXJuICgpID0+IHNldEZvbGRlclBhdGgoW10pXHJcbiAgfSwgW2N1cnJlbnRGb2xkZXIsIGZvbGRlclBhdGhbZm9sZGVyUGF0aC5sZW5ndGggLSAxXV0pIC8vIHdoZW4gdGhlIGxhc3QgZm9sZGVyIGluIHBhdGggaXMgY2hhbmdlZFxyXG5cclxuICBSZWFjdC51c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgY29uc3Qgam9pbmVkRm9sZGVyc0ZvclJlcXVlc3QgPSBmb2xkZXJQYXRoLmpvaW4oJy8nKS5zdWJzdHIoMSlcclxuICAgIGNvbnNvbGUubG9nKGpvaW5lZEZvbGRlcnNGb3JSZXF1ZXN0KVxyXG4gICAgc2V0T3ZlcmxhaWRQbG90cyh7IG5hbWU6ICcnLCBmb2xkZXJfcGF0aDogam9pbmVkRm9sZGVyc0ZvclJlcXVlc3QgfSlcclxuICB9LCBbZm9sZGVyUGF0aF0pXHJcblxyXG4gIGNvbnN0IGNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWIgPSAocGFyYW1ldGVyczogUGFyc2VkVXJsUXVlcnlJbnB1dCkgPT4ge1xyXG4gICAgY29uc29sZS5sb2cocGFyYW1ldGVycylcclxuICAgIGlmIChwYXJhbWV0ZXJzLmZvbGRlcl9wYXRoID09PSAnLycpIHtcclxuICAgICAgc2V0T3ZlcmxhaWRQbG90cyh7IGZvbGRlcl9wYXRoOiAnJywgbmFtZTogJycgfSk7XHJcbiAgICAgIHNldEZvbGRlclBhdGgoW10pXHJcbiAgICAgIHNldEN1cnJlbnRGb2xkZXIoJycpXHJcbiAgICB9XHJcbiAgICBjb25zdCBmb2xkZXJzID0gb3ZlcmxhaWRQbG90cy5mb2xkZXJfcGF0aC5zcGxpdCgnLycpXHJcbiAgICBzZXRGb2xkZXJQYXRoKGZvbGRlcnMpXHJcbiAgICBzZXRPdmVybGFpZFBsb3RzKHBhcmFtZXRlcnMpO1xyXG4gIH1cclxuXHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8TW9kYWxcclxuICAgICAgdmlzaWJsZT17dmlzaWJsZX1cclxuICAgICAgb25DYW5jZWw9eygpID0+IHtcclxuICAgICAgICBzZXRPcGVuT3ZlcmxheVdpdGhBbm90aGVyUGxvdE1vZGFsKGZhbHNlKVxyXG4gICAgICAgIHNldEZvbGRlclBhdGgoW10pXHJcbiAgICAgIH19XHJcbiAgICA+XHJcbiAgICAgIDxSb3cgZ3V0dGVyPXsxNn0+XHJcbiAgICAgICAgPENvbCBzdHlsZT17eyBwYWRkaW5nOiA4IH19PlxyXG4gICAgICAgICAgPEZvbGRlclBhdGggZm9sZGVyX3BhdGg9e292ZXJsYWlkUGxvdHMuZm9sZGVyX3BhdGh9IGNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWI9e2NoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWJ9IC8+XHJcbiAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgPFJvdyBzdHlsZT17eyB3aWR0aDogJzEwMCUnIH19PlxyXG4gICAgICAgICAge1xyXG4gICAgICAgICAgICBkYXRhLm1hcCgoZm9sZGVyX29yX3Bsb3Q6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICA8PlxyXG4gICAgICAgICAgICAgICAgICB7Zm9sZGVyX29yX3Bsb3Quc3ViZGlyICYmXHJcbiAgICAgICAgICAgICAgICAgICAgPENvbCBzcGFuPXs4fSBvbkNsaWNrPXsoKSA9PiBzZXRDdXJyZW50Rm9sZGVyKGZvbGRlcl9vcl9wbG90LnN1YmRpcil9PlxyXG4gICAgICAgICAgICAgICAgICAgICAgPEljb24gLz5cclxuICAgICAgICAgICAgICAgICAgICAgIDxTdHlsZWRBPntmb2xkZXJfb3JfcGxvdC5zdWJkaXJ9PC9TdHlsZWRBPlxyXG4gICAgICAgICAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICA8Lz5cclxuICAgICAgICAgICAgICApXHJcbiAgICAgICAgICAgIH0pXHJcbiAgICAgICAgICB9XHJcbiAgICAgICAgPC9Sb3c+XHJcbiAgICAgICAgPFJvdyBzdHlsZT17eyB3aWR0aDogJzEwMCUnIH19PlxyXG4gICAgICAgICAge1xyXG4gICAgICAgICAgICBkYXRhLm1hcCgoZm9sZGVyX29yX3Bsb3Q6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICA8PlxyXG4gICAgICAgICAgICAgICAgICB7Zm9sZGVyX29yX3Bsb3QubmFtZSAmJlxyXG4gICAgICAgICAgICAgICAgICAgIDxDb2wgc3Bhbj17OH0gb25DbGljaz17KCkgPT4gc2V0UGxvdChmb2xkZXJfb3JfcGxvdCl9PlxyXG4gICAgICAgICAgICAgICAgICAgICAgPEJ1dHRvbiA+e2ZvbGRlcl9vcl9wbG90Lm5hbWV9PC9CdXR0b24+XHJcbiAgICAgICAgICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIDwvPlxyXG4gICAgICAgICAgICAgIClcclxuICAgICAgICAgICAgfSlcclxuICAgICAgICAgIH1cclxuICAgICAgICA8L1Jvdz5cclxuICAgICAgPC9Sb3c+XHJcbiAgICA8L01vZGFsPlxyXG4gIClcclxufSJdLCJzb3VyY2VSb290IjoiIn0=