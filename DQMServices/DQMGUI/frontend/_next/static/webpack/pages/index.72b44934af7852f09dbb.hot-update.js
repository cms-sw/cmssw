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
      var joinderFolders = copy.join('/');
      setOverlaidPlots({
        folder_path: joinderFolders,
        name: ''
      });
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
      lineNumber: 71,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    gutter: 16,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 78,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
    style: {
      padding: 8
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 79,
      columnNumber: 9
    }
  }, __jsx(_containers_display_content_folderPath__WEBPACK_IMPORTED_MODULE_10__["FolderPath"], {
    folder_path: overlaidPlots.folder_path,
    changeFolderPathByBreadcrumb: changeFolderPathByBreadcrumb,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 80,
      columnNumber: 11
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    style: {
      width: '100%'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
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
        lineNumber: 88,
        columnNumber: 21
      }
    }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["Icon"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 89,
        columnNumber: 23
      }
    }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledA"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 90,
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
      lineNumber: 98,
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
        lineNumber: 104,
        columnNumber: 21
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Button"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 105,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9vdmVybGF5V2l0aEFub3RoZXJQbG90L2luZGV4LnRzeCJdLCJuYW1lcyI6WyJPdmVybGF5V2l0aEFub3RoZXJQbG90IiwidmlzaWJsZSIsInNldE9wZW5PdmVybGF5V2l0aEFub3RoZXJQbG90TW9kYWwiLCJSZWFjdCIsImZvbGRlcl9wYXRoIiwibmFtZSIsIm92ZXJsYWlkUGxvdHMiLCJzZXRPdmVybGFpZFBsb3RzIiwiZm9sZGVyUGF0aCIsInNldEZvbGRlclBhdGgiLCJmb2xkZXJzIiwic2V0Rm9sZGVycyIsImN1cnJlbnRGb2xkZXIiLCJzZXRDdXJyZW50Rm9sZGVyIiwicGxvdCIsInNldFBsb3QiLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJxdWVyeSIsInN0b3JlIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsInBhcmFtcyIsImRhdGFzZXRfbmFtZSIsInJ1bl9udW1iZXIiLCJub3RPbGRlclRoYW4iLCJmb2xkZXJzX3BhdGgiLCJwbG90X25hbWUiLCJhcGkiLCJjaG9vc2VfYXBpIiwiZGF0YV9nZXRfYnlfbW91bnQiLCJ1c2VSZXF1ZXN0IiwiY29weSIsImluZGV4IiwiaW5kZXhPZiIsImZpbHRlciIsImZvbGRlciIsInB1c2giLCJqb2luZGVyRm9sZGVycyIsImpvaW4iLCJkYXRhIiwiZm9sZGVyc19vcl9wbG90cyIsImNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWIiLCJpdGVtIiwiZm9sZGVyc19mcm9tX2JyZWFkY3J1bWIiLCJzcGxpdCIsImxlbmd0aCIsImNvbnNvbGUiLCJsb2ciLCJwYWRkaW5nIiwid2lkdGgiLCJtYXAiLCJmb2xkZXJfb3JfcGxvdCIsInN1YmRpciJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUdBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQVFPLElBQU1BLHNCQUFzQixHQUFHLFNBQXpC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QUEsT0FDRSxNQUFDLHlFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFERixFQUVFLE1BQUMsNEVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUFVRCxjQUFjLENBQUNDLE1BQXpCLENBRkYsQ0FGSixDQURGO0FBVUQsR0FYRCxDQUZKLENBSkYsRUFvQkUsTUFBQyx3Q0FBRDtBQUFLLFNBQUssRUFBRTtBQUFFSCxXQUFLLEVBQUU7QUFBVCxLQUFaO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FFSVQsZ0JBQWdCLENBQUNVLEdBQWpCLENBQXFCLFVBQUNDLGNBQUQsRUFBeUI7QUFDNUMsV0FDRSw0REFDR0EsY0FBYyxDQUFDOUMsSUFBZixJQUNDLE1BQUMsd0NBQUQ7QUFBSyxVQUFJLEVBQUUsQ0FBWDtBQUFjLGFBQU8sRUFBRTtBQUFBLGVBQU1VLE9BQU8sQ0FBQ29DLGNBQUQsQ0FBYjtBQUFBLE9BQXZCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDRSxNQUFDLDJDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FBVUEsY0FBYyxDQUFDOUMsSUFBekIsQ0FERixDQUZKLENBREY7QUFTRCxHQVZELENBRkosQ0FwQkYsQ0FQRixDQURGO0FBOENELENBakdNOztHQUFNTCxzQjtVQU9JaUIscUQsRUFhV2EsNEQ7OztLQXBCZjlCLHNCIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjcyYjQ0OTM0YWY3ODUyZjA5ZGJiLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCdcclxuaW1wb3J0IE1vZGFsIGZyb20gJ2FudGQvbGliL21vZGFsL01vZGFsJ1xyXG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcidcclxuXHJcbmltcG9ydCB7IFBhcmFtc0ZvckFwaVByb3BzLCBQbG90b3ZlcmxhaWRTZXBhcmF0ZWx5UHJvcHMsIFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcydcclxuaW1wb3J0IHsgSWNvbiwgU3R5bGVkQSB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJ1xyXG5pbXBvcnQgeyBjaG9vc2VfYXBpIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3V0aWxzJ1xyXG5pbXBvcnQgeyBzdG9yZSB9IGZyb20gJy4uLy4uL2NvbnRleHRzL2xlZnRTaWRlQ29udGV4dCdcclxuaW1wb3J0IHsgdXNlUmVxdWVzdCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVJlcXVlc3QnXHJcbmltcG9ydCB7IEJ1dHRvbiwgQ29sLCBSb3cgfSBmcm9tICdhbnRkJ1xyXG5pbXBvcnQgeyBGb2xkZXJQYXRoIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2NvbnRlbnQvZm9sZGVyUGF0aCdcclxuaW1wb3J0IHsgUGFyc2VkVXJsUXVlcnlJbnB1dCB9IGZyb20gJ3F1ZXJ5c3RyaW5nJ1xyXG5cclxuaW50ZXJmYWNlIE92ZXJsYXlXaXRoQW5vdGhlclBsb3RQcm9wcyB7XHJcbiAgdmlzaWJsZTogYm9vbGVhbjtcclxuICBzZXRPcGVuT3ZlcmxheVdpdGhBbm90aGVyUGxvdE1vZGFsOiBhbnlcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IE92ZXJsYXlXaXRoQW5vdGhlclBsb3QgPSAoeyB2aXNpYmxlLCBzZXRPcGVuT3ZlcmxheVdpdGhBbm90aGVyUGxvdE1vZGFsIH06IE92ZXJsYXlXaXRoQW5vdGhlclBsb3RQcm9wcykgPT4ge1xyXG4gIGNvbnN0IFtvdmVybGFpZFBsb3RzLCBzZXRPdmVybGFpZFBsb3RzXSA9IFJlYWN0LnVzZVN0YXRlPFBsb3RvdmVybGFpZFNlcGFyYXRlbHlQcm9wcz4oeyBmb2xkZXJfcGF0aDogJycsIG5hbWU6ICcnIH0pXHJcbiAgY29uc3QgW2ZvbGRlclBhdGgsIHNldEZvbGRlclBhdGhdID0gUmVhY3QudXNlU3RhdGU8c3RyaW5nW10+KCcnKVxyXG4gIGNvbnN0IFtmb2xkZXJzLCBzZXRGb2xkZXJzXSA9IFJlYWN0LnVzZVN0YXRlPHN0cmluZ1tdPihbXSlcclxuICBjb25zdCBbY3VycmVudEZvbGRlciwgc2V0Q3VycmVudEZvbGRlcl0gPSBSZWFjdC51c2VTdGF0ZSgnJylcclxuICBjb25zdCBbcGxvdCwgc2V0UGxvdF0gPSBSZWFjdC51c2VTdGF0ZSh7fSlcclxuXHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcbiAgY29uc3QgeyB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIH0gPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKVxyXG5cclxuICBjb25zdCBwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzID0ge1xyXG4gICAgZGF0YXNldF9uYW1lOiBxdWVyeS5kYXRhc2V0X25hbWUgYXMgc3RyaW5nLFxyXG4gICAgcnVuX251bWJlcjogcXVlcnkucnVuX251bWJlciBhcyBzdHJpbmcsXHJcbiAgICBub3RPbGRlclRoYW46IHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4sXHJcbiAgICBmb2xkZXJzX3BhdGg6IG92ZXJsYWlkUGxvdHMuZm9sZGVyX3BhdGgsXHJcbiAgICBwbG90X25hbWU6IG92ZXJsYWlkUGxvdHMubmFtZVxyXG4gIH1cclxuXHJcbiAgY29uc3QgYXBpID0gY2hvb3NlX2FwaShwYXJhbXMpXHJcbiAgY29uc3QgZGF0YV9nZXRfYnlfbW91bnQgPSB1c2VSZXF1ZXN0KGFwaSxcclxuICAgIHt9LFxyXG4gICAgW292ZXJsYWlkUGxvdHMuZm9sZGVyX3BhdGhdXHJcbiAgKTtcclxuXHJcbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGNvbnN0IGNvcHkgPSBbLi4uZm9sZGVyc11cclxuICAgIGNvbnN0IGluZGV4ID0gZm9sZGVycy5pbmRleE9mKGN1cnJlbnRGb2xkZXIpXHJcblxyXG4gICAgaWYgKGluZGV4ID49IDApIHtcclxuICAgICAgc2V0Rm9sZGVycyhmb2xkZXJzLmZpbHRlcigoZm9sZGVyOiBzdHJpbmcpID0+IGZvbGRlciAhPT0gZm9sZGVyc1tpbmRleF0pKVxyXG4gICAgfVxyXG4gICAgZWxzZSB7XHJcbiAgICAgIGNvcHkucHVzaChjdXJyZW50Rm9sZGVyKVxyXG4gICAgICBzZXRGb2xkZXJzKGNvcHkpXHJcbiAgICAgIGNvbnN0IGpvaW5kZXJGb2xkZXJzID0gY29weS5qb2luKCcvJylcclxuICAgICAgc2V0T3ZlcmxhaWRQbG90cyh7Zm9sZGVyX3BhdGg6IGpvaW5kZXJGb2xkZXJzLCBuYW1lOiAnJ30pXHJcbiAgICB9XHJcbiAgfSwgW2N1cnJlbnRGb2xkZXJdKVxyXG5cclxuICBjb25zdCB7IGRhdGEgfSA9IGRhdGFfZ2V0X2J5X21vdW50XHJcbiAgY29uc3QgZm9sZGVyc19vcl9wbG90cyA9IGRhdGEgPyBkYXRhLmRhdGEgOiBbXVxyXG5cclxuICBjb25zdCBjaGFuZ2VGb2xkZXJQYXRoQnlCcmVhZGNydW1iID0gKGl0ZW0pID0+IHtcclxuICAgIGNvbnN0IGZvbGRlcnNfZnJvbV9icmVhZGNydW1iID0gaXRlbS5mb2xkZXJfcGF0aC5zcGxpdCgnLycpXHJcbiAgICBzZXRDdXJyZW50Rm9sZGVyKGZvbGRlcnNfZnJvbV9icmVhZGNydW1iW2ZvbGRlcnNfZnJvbV9icmVhZGNydW1iLmxlbmd0aCAtIDFdKVxyXG4gICAgc2V0Rm9sZGVycyhmb2xkZXJzX2Zyb21fYnJlYWRjcnVtYilcclxuICB9XHJcblxyXG5cclxuICBjb25zb2xlLmxvZyhjdXJyZW50Rm9sZGVyKVxyXG4gIHJldHVybiAoXHJcbiAgICA8TW9kYWxcclxuICAgICAgdmlzaWJsZT17dmlzaWJsZX1cclxuICAgICAgb25DYW5jZWw9eygpID0+IHtcclxuICAgICAgICBzZXRPcGVuT3ZlcmxheVdpdGhBbm90aGVyUGxvdE1vZGFsKGZhbHNlKVxyXG4gICAgICAgIHNldEZvbGRlclBhdGgoW10pXHJcbiAgICAgIH19XHJcbiAgICA+XHJcbiAgICAgIDxSb3cgZ3V0dGVyPXsxNn0+XHJcbiAgICAgICAgPENvbCBzdHlsZT17eyBwYWRkaW5nOiA4IH19PlxyXG4gICAgICAgICAgPEZvbGRlclBhdGggZm9sZGVyX3BhdGg9e292ZXJsYWlkUGxvdHMuZm9sZGVyX3BhdGh9IGNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWI9e2NoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWJ9IC8+XHJcbiAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgPFJvdyBzdHlsZT17eyB3aWR0aDogJzEwMCUnIH19PlxyXG4gICAgICAgICAge1xyXG4gICAgICAgICAgICBmb2xkZXJzX29yX3Bsb3RzLm1hcCgoZm9sZGVyX29yX3Bsb3Q6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICA8PlxyXG4gICAgICAgICAgICAgICAgICB7Zm9sZGVyX29yX3Bsb3Quc3ViZGlyICYmXHJcbiAgICAgICAgICAgICAgICAgICAgPENvbCBzcGFuPXs4fSBvbkNsaWNrPXsoKSA9PiBzZXRDdXJyZW50Rm9sZGVyKGZvbGRlcl9vcl9wbG90LnN1YmRpcil9PlxyXG4gICAgICAgICAgICAgICAgICAgICAgPEljb24gLz5cclxuICAgICAgICAgICAgICAgICAgICAgIDxTdHlsZWRBPntmb2xkZXJfb3JfcGxvdC5zdWJkaXJ9PC9TdHlsZWRBPlxyXG4gICAgICAgICAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICA8Lz5cclxuICAgICAgICAgICAgICApXHJcbiAgICAgICAgICAgIH0pXHJcbiAgICAgICAgICB9XHJcbiAgICAgICAgPC9Sb3c+XHJcbiAgICAgICAgPFJvdyBzdHlsZT17eyB3aWR0aDogJzEwMCUnIH19PlxyXG4gICAgICAgICAge1xyXG4gICAgICAgICAgICBmb2xkZXJzX29yX3Bsb3RzLm1hcCgoZm9sZGVyX29yX3Bsb3Q6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICA8PlxyXG4gICAgICAgICAgICAgICAgICB7Zm9sZGVyX29yX3Bsb3QubmFtZSAmJlxyXG4gICAgICAgICAgICAgICAgICAgIDxDb2wgc3Bhbj17OH0gb25DbGljaz17KCkgPT4gc2V0UGxvdChmb2xkZXJfb3JfcGxvdCl9PlxyXG4gICAgICAgICAgICAgICAgICAgICAgPEJ1dHRvbiA+e2ZvbGRlcl9vcl9wbG90Lm5hbWV9PC9CdXR0b24+XHJcbiAgICAgICAgICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIDwvPlxyXG4gICAgICAgICAgICAgIClcclxuICAgICAgICAgICAgfSlcclxuICAgICAgICAgIH1cclxuICAgICAgICA8L1Jvdz5cclxuICAgICAgPC9Sb3c+XHJcbiAgICA8L01vZGFsPlxyXG4gIClcclxufSJdLCJzb3VyY2VSb290IjoiIn0=